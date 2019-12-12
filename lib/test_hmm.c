// SPDX-License-Identifier: GPL-2.0
/*
 * This is a module to test the HMM (Heterogeneous Memory Management)
 * mirror and zone device private memory migration APIs of the kernel.
 * Userspace programs can register with the driver to mirror their own address
 * space and can use the device to read/write any valid virtual address.
 */
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/mutex.h>
#include <linux/rwsem.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/highmem.h>
#include <linux/delay.h>
#include <linux/pagemap.h>
#include <linux/hmm.h>
#include <linux/vmalloc.h>
#include <linux/swap.h>
#include <linux/swapops.h>
#include <linux/sched/mm.h>
#include <linux/platform_device.h>

#include <uapi/linux/test_hmm.h>

#define DMIRROR_NDEVICES		2
#define DMIRROR_RANGE_FAULT_TIMEOUT	1000
#define DEVMEM_CHUNK_SIZE		(256 * 1024 * 1024U)
#define DEVMEM_CHUNKS_RESERVE		16

static const struct dev_pagemap_ops dmirror_devmem_ops;
static const struct mmu_interval_notifier_ops dmirror_min_ops;
static dev_t dmirror_dev;
static struct page *dmirror_zero_page;

struct dmirror_device;

struct dmirror_bounce {
	void			*ptr;
	unsigned long		size;
	unsigned long		addr;
	unsigned long		cpages;
};

#define DPT_SHIFT PAGE_SHIFT
#define DPT_VALID (1UL << 0)
#define DPT_WRITE (1UL << 1)
#define DPT_DPAGE (1UL << 2)

#define DPT_XA_TAG_WRITE 3UL

static const uint64_t dmirror_hmm_flags[HMM_PFN_FLAG_MAX] = {
	[HMM_PFN_VALID] = DPT_VALID,
	[HMM_PFN_WRITE] = DPT_WRITE,
	[HMM_PFN_DEVICE_PRIVATE] = DPT_DPAGE,
};

static const uint64_t dmirror_hmm_values[HMM_PFN_VALUE_MAX] = {
	[HMM_PFN_NONE]    = 0,
	[HMM_PFN_ERROR]   = 0x10,
	[HMM_PFN_SPECIAL] = 0x10,
};

/*
 * Data structure to track address ranges and register for mmu interval
 * notifier updates.
 */
struct dmirror_interval {
	struct mmu_interval_notifier	notifier;
	struct dmirror			*dmirror;
};

/*
 * Data attached to the open device file.
 * Note that it might be shared after a fork().
 */
struct dmirror {
	struct mm_struct	*mm;
	struct dmirror_device	*mdevice;
	struct xarray		pt;
	struct mutex		mutex;
};

/*
 * ZONE_DEVICE pages for migration and simulating device memory.
 */
struct dmirror_chunk {
	struct dev_pagemap	pagemap;
	struct dmirror_device	*mdevice;
};

/*
 * Per device data.
 */
struct dmirror_device {
	struct cdev		cdevice;
	struct hmm_devmem	*devmem;

	unsigned int		devmem_capacity;
	unsigned int		devmem_count;
	struct dmirror_chunk	**devmem_chunks;
	struct mutex		devmem_lock;	/* protects the above */

	unsigned long		calloc;
	unsigned long		cfree;
	struct page		*free_pages;
	struct page		*free_huge_pages;
	spinlock_t		lock;		/* protects the above */
};

static struct dmirror_device dmirror_devices[DMIRROR_NDEVICES];

static int dmirror_bounce_init(struct dmirror_bounce *bounce,
			       unsigned long addr,
			       unsigned long size)
{
	bounce->addr = addr;
	bounce->size = size;
	bounce->cpages = 0;
	bounce->ptr = vmalloc(size);
	if (!bounce->ptr)
		return -ENOMEM;
	return 0;
}

static void dmirror_bounce_fini(struct dmirror_bounce *bounce)
{
	vfree(bounce->ptr);
}

static int dmirror_fops_open(struct inode *inode, struct file *filp)
{
	struct cdev *cdev = inode->i_cdev;
	struct dmirror *dmirror;
	int ret;

	/* Mirror this process address space */
	dmirror = kzalloc(sizeof(*dmirror), GFP_KERNEL);
	if (dmirror == NULL)
		return -ENOMEM;

	dmirror->mdevice = container_of(cdev, struct dmirror_device, cdevice);
	mutex_init(&dmirror->mutex);
	xa_init(&dmirror->pt);

	/*
	 * Pre-register for mmu interval notifiers so
	 * mmu_interval_notifier_insert_safe() can be called without holding
	 * mmap_sem for write.
	 */
	ret = mmu_notifier_register(NULL, current->mm);
	if (ret) {
		kfree(dmirror);
		return ret;
	}

	/* Pairs with the mmdrop() in dmirror_fops_release(). */
	mmgrab(current->mm);
	dmirror->mm = current->mm;

	/* Only the first open registers the address space. */
	filp->private_data = dmirror;
	return ret;
}

static int dmirror_fops_release(struct inode *inode, struct file *filp)
{
	struct dmirror *dmirror = filp->private_data;
	struct mmu_interval_notifier *mni;

	mutex_lock(&dmirror->mutex);
	while (true) {
		mni = mmu_interval_notifier_find(dmirror->mm, &dmirror_min_ops,
						 0UL, ~0UL);
		if (!mni)
			break;
		mmu_interval_notifier_remove_deferred(mni);
	}
	mutex_unlock(&dmirror->mutex);
	mmu_interval_notifier_synchronize(dmirror->mm);
	mmdrop(dmirror->mm);
	xa_destroy(&dmirror->pt);
	kfree(dmirror);
	return 0;
}

static inline struct dmirror_device *dmirror_page_to_device(struct page *page)

{
	struct dmirror_chunk *devmem;

	devmem = container_of(page->pgmap, struct dmirror_chunk, pagemap);
	return devmem->mdevice;
}

static bool dmirror_device_is_mine(struct dmirror_device *mdevice,
				   struct page *page)
{
	if (!is_zone_device_page(page))
		return false;
	return page->pgmap->ops == &dmirror_devmem_ops &&
		dmirror_page_to_device(page) == mdevice;
}

static int dmirror_do_fault(struct dmirror *dmirror, struct hmm_range *range)
{
	uint64_t *pfns = range->pfns;
	unsigned long pfn;

	for (pfn = (range->start >> PAGE_SHIFT);
	     pfn < (range->end >> PAGE_SHIFT);
	     pfn++, pfns++) {
		struct page *page;
		void *entry;

		/*
		 * HMM_PFN_ERROR is returned if it is accessing invalid memory
		 * either because of memory error (hardware detected memory
		 * corruption) or more likely because of truncate on mmap
		 * file.
		 */
		if (*pfns == range->values[HMM_PFN_ERROR])
			return -EFAULT;
		if (!(*pfns & range->flags[HMM_PFN_VALID]))
			return -EFAULT;
		page = hmm_device_entry_to_page(range, *pfns);
		/* We asked for pages to be populated but check anyway. */
		if (!page)
			return -EFAULT;
		if (is_zone_device_page(page)) {
			/*
			 * TODO: need a way to ask HMM to fault foreign zone
			 * device private pages.
			 */
			if (!dmirror_device_is_mine(dmirror->mdevice, page))
				continue;
		}
		entry = page;
		if (*pfns & range->flags[HMM_PFN_WRITE])
			entry = xa_tag_pointer(entry, DPT_XA_TAG_WRITE);
		else if (range->default_flags & range->flags[HMM_PFN_WRITE])
			return -EFAULT;
		entry = xa_store(&dmirror->pt, pfn, entry, GFP_ATOMIC);
		if (xa_is_err(entry))
			return xa_err(entry);
	}

	return 0;
}

static void dmirror_do_update(struct dmirror *dmirror, unsigned long start,
			      unsigned long end)
{
	unsigned long pfn;

	/*
	 * The XArray doesn't hold references to pages since it relies on
	 * the mmu notifier to clear pointers when they become stale.
	 * Therefore, it is OK to just clear the entry.
	 */
	for (pfn = start >> PAGE_SHIFT; pfn < (end >> PAGE_SHIFT); pfn++)
		xa_erase(&dmirror->pt, pfn);
}

static struct dmirror_interval *dmirror_new_interval(struct dmirror *dmirror,
						     unsigned long start,
						     unsigned long last)
{
	struct dmirror_interval *dmi;
	int ret;

	dmi = kmalloc(sizeof(*dmi), GFP_ATOMIC);
	if (!dmi)
		return NULL;

	dmi->dmirror = dmirror;

	ret = mmu_interval_notifier_insert_safe(&dmi->notifier, dmirror->mm,
				start, last - start + 1, &dmirror_min_ops);
	if (ret) {
		kfree(dmi);
		return NULL;
	}

	return dmi;
}

static void dmirror_do_unmap(struct mmu_interval_notifier *mni,
			     const struct mmu_notifier_range *range)
{
	struct dmirror_interval *dmi =
		container_of(mni, struct dmirror_interval, notifier);
	struct dmirror *dmirror = dmi->dmirror;
	unsigned long start = mmu_interval_notifier_start(mni);
	unsigned long last = mmu_interval_notifier_last(mni);

	if (start >= range->start) {
		/* Remove the whole interval or keep the right-hand part. */
		if (last <= range->end)
			mmu_interval_notifier_remove_deferred(mni);
		else
			mmu_interval_notifier_update(mni, range->end, last);
		return;
	}

	/* Keep the left-hand part of the interval. */
	mmu_interval_notifier_update(mni, start, range->start - 1);

	/*
	 * If a hole is created, create an interval for the right-hand part.
	 * Note that the two operations, shrink existing interval and
	 * insert new inverval, will be atomic as seen by the core mmu
	 * interval code because they will be deferred until the last task
	 * walking the interval tree completes and these operations are
	 * processed while holding the lock protecting changes to the
	 * interval tree (see mn_itree_inv_end()).
	 * If these two operations were done outside of the invalidate()
	 * callback, there would be a window where an invalidation could be
	 * missed because the shrink would be visible but not the insertion.
	 */
	if (last >= range->end) {
		dmi = dmirror_new_interval(dmirror, range->end, last);
		/*
		 * If we can't allocate an interval, we won't get invalidation
		 * callbacks so clear the mapping and rely on faults to reload
		 * the mappings if needed.
		 */
		if (!dmi)
			dmirror_do_update(dmirror, range->end, last + 1);
	}
}

static bool dmirror_interval_invalidate(struct mmu_interval_notifier *mni,
				const struct mmu_notifier_range *range,
				unsigned long cur_seq)
{
	struct dmirror_interval *dmi =
		container_of(mni, struct dmirror_interval, notifier);
	struct dmirror *dmirror = dmi->dmirror;
	unsigned long start = mmu_interval_notifier_start(mni);
	unsigned long last = mmu_interval_notifier_last(mni);

	if (mmu_notifier_range_blockable(range))
		mutex_lock(&dmirror->mutex);
	else if (!mutex_trylock(&dmirror->mutex))
		return false;

	mmu_interval_set_seq(mni, cur_seq);
	dmirror_do_update(dmirror, max(start, range->start),
			  min(last + 1, range->end));

	/* Stop tracking the range if it is an unmap. */
	if (range->event == MMU_NOTIFY_UNMAP)
		dmirror_do_unmap(mni, range);
	else if (range->event == MMU_NOTIFY_RELEASE)
		mmu_interval_notifier_remove_deferred(mni);

	mutex_unlock(&dmirror->mutex);
	return true;
}

static void dmirror_interval_release(struct mmu_interval_notifier *mni)
{
	struct dmirror_interval *dmi =
		container_of(mni, struct dmirror_interval, notifier);

	kfree(dmi);
}

static const struct mmu_interval_notifier_ops dmirror_min_ops = {
	.invalidate = dmirror_interval_invalidate,
	.release = dmirror_interval_release,
};

/*
 * Find or create a mmu_interval_notifier for the given range.
 * Although mmu_interval_notifier_insert_safe() can handle overlapping
 * intervals, we only create non-overlapping intervals, shrinking the hmm_range
 * if it spans more than one dmirror_interval.
 */
static int dmirror_interval_find(struct dmirror *dmirror,
				 struct hmm_range *range)
{
	struct mmu_interval_notifier *mni;
	struct dmirror_interval *dmi;
	struct vm_area_struct *vma;
	unsigned long start = range->start;
	unsigned long last = range->end - 1;
	int ret;

	mutex_lock(&dmirror->mutex);
	mni = mmu_interval_notifier_find(dmirror->mm, &dmirror_min_ops, start,
					 last);
	if (mni) {
		if (start >= mmu_interval_notifier_start(mni)) {
			dmi = container_of(mni, struct dmirror_interval,
					   notifier);
			if (last > mmu_interval_notifier_last(mni))
				range->end =
					mmu_interval_notifier_last(mni) + 1;
			goto found;
		}
		WARN_ON(last <= mmu_interval_notifier_start(mni));
		range->end = mmu_interval_notifier_start(mni);
		last = range->end - 1;
	}
	/*
	 * Might as well create an interval covering the underlying VMA to
	 * avoid having to create a bunch of small intervals.
	 */
	vma = find_vma(dmirror->mm, start);
	if (!vma || start < vma->vm_start) {
		ret = -ENOENT;
		goto err;
	}
	if (range->end > vma->vm_end) {
		range->end = vma->vm_end;
		last = range->end - 1;
	} else if (!mni) {
		/* Anything registered on the right part of the vma? */
		mni = mmu_interval_notifier_find(dmirror->mm, &dmirror_min_ops,
						 range->end, vma->vm_end - 1);
		if (mni)
			last = mmu_interval_notifier_start(mni) - 1;
		else
			last = vma->vm_end - 1;
	}
	/* Anything registered on the left part of the vma? */
	mni = mmu_interval_notifier_find(dmirror->mm, &dmirror_min_ops,
					 vma->vm_start, start - 1);
	if (mni)
		start = mmu_interval_notifier_last(mni) + 1;
	else
		start = vma->vm_start;
	dmi = dmirror_new_interval(dmirror, start, last);
	if (!dmi) {
		ret = -ENOMEM;
		goto err;
	}

found:
	range->notifier = &dmi->notifier;
	mutex_unlock(&dmirror->mutex);
	return 0;

err:
	mutex_unlock(&dmirror->mutex);
	return ret;
}

static int dmirror_range_fault(struct dmirror *dmirror,
				struct hmm_range *range)
{
	struct mm_struct *mm = dmirror->mm;
	unsigned long timeout =
		jiffies + msecs_to_jiffies(HMM_RANGE_DEFAULT_TIMEOUT);
	int ret;

	while (true) {
		long count;

		if (time_after(jiffies, timeout)) {
			ret = -EBUSY;
			goto out;
		}

		down_read(&mm->mmap_sem);
		ret = dmirror_interval_find(dmirror, range);
		if (ret) {
			up_read(&mm->mmap_sem);
			goto out;
		}
		range->notifier_seq = mmu_interval_read_begin(range->notifier);
		count = hmm_range_fault(range, 0);
		up_read(&mm->mmap_sem);
		if (count <= 0) {
			if (count == 0 || count == -EBUSY)
				continue;
			ret = count;
			goto out;
		}

		mutex_lock(&dmirror->mutex);
		if (mmu_interval_read_retry(range->notifier,
					    range->notifier_seq)) {
			mutex_unlock(&dmirror->mutex);
			continue;
		}
		break;
	}

	ret = dmirror_do_fault(dmirror, range);

	mutex_unlock(&dmirror->mutex);
out:
	return ret;
}

static int dmirror_fault(struct dmirror *dmirror, unsigned long start,
			 unsigned long end, bool write)
{
	struct mm_struct *mm = dmirror->mm;
	unsigned long addr;
	unsigned long next;
	uint64_t pfns[64];
	struct hmm_range range = {
		.pfns = pfns,
		.flags = dmirror_hmm_flags,
		.values = dmirror_hmm_values,
		.pfn_shift = DPT_SHIFT,
		.pfn_flags_mask = ~(dmirror_hmm_flags[HMM_PFN_VALID] |
				    dmirror_hmm_flags[HMM_PFN_WRITE]),
		.default_flags = dmirror_hmm_flags[HMM_PFN_VALID] |
				(write ? dmirror_hmm_flags[HMM_PFN_WRITE] : 0),
	};
	int ret = 0;

	/* Since the mm is for the mirrored process, get a reference first. */
	if (!mmget_not_zero(mm))
		return 0;

	for (addr = start; addr < end; addr = next) {
		next = min(addr + (ARRAY_SIZE(pfns) << PAGE_SHIFT), end);
		range.start = addr;
		range.end = next;

		ret = dmirror_range_fault(dmirror, &range);
		if (ret)
			break;
	}

	mmput(mm);
	return ret;
}

static int dmirror_do_read(struct dmirror *dmirror, unsigned long start,
			   unsigned long end, struct dmirror_bounce *bounce)
{
	unsigned long pfn;
	void *ptr;

	ptr = bounce->ptr + ((start - bounce->addr) & PAGE_MASK);

	for (pfn = start >> PAGE_SHIFT; pfn < (end >> PAGE_SHIFT); pfn++) {
		void *entry;
		struct page *page;
		void *tmp;

		entry = xa_load(&dmirror->pt, pfn);
		page = xa_untag_pointer(entry);
		if (!page)
			return -ENOENT;

		tmp = kmap(page);
		memcpy(ptr, tmp, PAGE_SIZE);
		kunmap(page);

		ptr += PAGE_SIZE;
		bounce->cpages++;
	}

	return 0;
}

static int dmirror_read(struct dmirror *dmirror, struct hmm_dmirror_cmd *cmd)
{
	struct dmirror_bounce bounce;
	unsigned long start, end;
	unsigned long size = cmd->npages << PAGE_SHIFT;
	int ret;

	start = cmd->addr;
	end = start + size;
	if (end < start)
		return -EINVAL;

	ret = dmirror_bounce_init(&bounce, start, size);
	if (ret)
		return ret;

again:
	mutex_lock(&dmirror->mutex);
	ret = dmirror_do_read(dmirror, start, end, &bounce);
	mutex_unlock(&dmirror->mutex);
	if (ret == 0)
		ret = copy_to_user((void __user *)cmd->ptr, bounce.ptr,
					bounce.size);
	else if (ret == -ENOENT) {
		start = cmd->addr + (bounce.cpages << PAGE_SHIFT);
		ret = dmirror_fault(dmirror, start, end, false);
		if (ret == 0) {
			cmd->faults++;
			goto again;
		}
	}

	cmd->cpages = bounce.cpages;
	dmirror_bounce_fini(&bounce);
	return ret;
}

static int dmirror_do_write(struct dmirror *dmirror, unsigned long start,
			    unsigned long end, struct dmirror_bounce *bounce)
{
	unsigned long pfn;
	void *ptr;

	ptr = bounce->ptr + ((start - bounce->addr) & PAGE_MASK);

	for (pfn = start >> PAGE_SHIFT; pfn < (end >> PAGE_SHIFT); pfn++) {
		void *entry;
		struct page *page;
		void *tmp;

		entry = xa_load(&dmirror->pt, pfn);
		page = xa_untag_pointer(entry);
		if (!page || xa_pointer_tag(entry) != DPT_XA_TAG_WRITE)
			return -ENOENT;

		tmp = kmap(page);
		memcpy(tmp, ptr, PAGE_SIZE);
		kunmap(page);

		ptr += PAGE_SIZE;
		bounce->cpages++;
	}

	return 0;
}

static int dmirror_write(struct dmirror *dmirror, struct hmm_dmirror_cmd *cmd)
{
	struct dmirror_bounce bounce;
	unsigned long start, end;
	unsigned long size = cmd->npages << PAGE_SHIFT;
	int ret;

	start = cmd->addr;
	end = start + size;
	if (end < start)
		return -EINVAL;

	ret = dmirror_bounce_init(&bounce, start, size);
	if (ret)
		return ret;
	ret = copy_from_user(bounce.ptr, (void __user *)cmd->ptr,
				bounce.size);
	if (ret)
		return ret;

again:
	mutex_lock(&dmirror->mutex);
	ret = dmirror_do_write(dmirror, start, end, &bounce);
	mutex_unlock(&dmirror->mutex);
	if (ret == -ENOENT) {
		start = cmd->addr + (bounce.cpages << PAGE_SHIFT);
		ret = dmirror_fault(dmirror, start, end, true);
		if (ret == 0) {
			cmd->faults++;
			goto again;
		}
	}

	cmd->cpages = bounce.cpages;
	dmirror_bounce_fini(&bounce);
	return ret;
}

static bool dmirror_allocate_chunk(struct dmirror_device *mdevice,
				   bool is_huge,
				   struct page **ppage)
{
	struct dmirror_chunk *devmem;
	struct resource *res;
	unsigned long pfn;
	unsigned long pfn_first;
	unsigned long pfn_last;
	void *ptr;

	mutex_lock(&mdevice->devmem_lock);

	if (mdevice->devmem_count == mdevice->devmem_capacity) {
		struct dmirror_chunk **new_chunks;
		unsigned int new_capacity;

		new_capacity = mdevice->devmem_capacity +
				DEVMEM_CHUNKS_RESERVE;
		new_chunks = krealloc(mdevice->devmem_chunks,
				sizeof(new_chunks[0]) * new_capacity,
				GFP_KERNEL);
		if (!new_chunks)
			goto err;
		mdevice->devmem_capacity = new_capacity;
		mdevice->devmem_chunks = new_chunks;
	}

	res = request_free_mem_region(&iomem_resource, DEVMEM_CHUNK_SIZE,
					"hmm_dmirror");
	if (IS_ERR(res))
		goto err;

	devmem = kzalloc(sizeof(*devmem), GFP_KERNEL);
	if (!devmem)
		goto err;

	devmem->pagemap.type = MEMORY_DEVICE_PRIVATE;
	devmem->pagemap.res = *res;
	devmem->pagemap.ops = &dmirror_devmem_ops;

	ptr = memremap_pages(&devmem->pagemap, numa_node_id());
	if (IS_ERR(ptr))
		goto err_free;

	devmem->mdevice = mdevice;
	pfn_first = devmem->pagemap.res.start >> PAGE_SHIFT;
	pfn_last = pfn_first +
		(resource_size(&devmem->pagemap.res) >> PAGE_SHIFT);
	mdevice->devmem_chunks[mdevice->devmem_count++] = devmem;

	mutex_unlock(&mdevice->devmem_lock);

	pr_info("added new %u MB chunk (total %u chunks, %u MB) PFNs [0x%lx 0x%lx)\n",
		DEVMEM_CHUNK_SIZE / (1024 * 1024),
		mdevice->devmem_count,
		mdevice->devmem_count * (DEVMEM_CHUNK_SIZE / (1024 * 1024)),
		pfn_first, pfn_last);

	spin_lock(&mdevice->lock);
	for (pfn = pfn_first; pfn < pfn_last; ) {
		struct page *page = pfn_to_page(pfn);

#ifdef CONFIG_TRANSPARENT_HUGEPAGE
		/*
		 * Check for PMD aligned PFN and create a huge page.
		 * Check for "< pfn_last - 1" so that the last two huge pages
		 * are used for normal pages.
		 */
		if ((pfn & (HPAGE_PMD_NR - 1)) == 0 &&
		    pfn + HPAGE_PMD_NR < pfn_last - 1) {
			prep_compound_page(page, HPAGE_PMD_ORDER);
			page->zone_device_data = mdevice->free_huge_pages;
			mdevice->free_huge_pages = page;
			pfn += HPAGE_PMD_NR;
			percpu_ref_put_many(page->pgmap->ref, HPAGE_PMD_NR - 1);
			continue;
		}
#endif
		page->zone_device_data = mdevice->free_pages;
		mdevice->free_pages = page;
		pfn++;
	}
	if (ppage) {
		if (is_huge) {
			*ppage = mdevice->free_huge_pages;
			mdevice->free_huge_pages = (*ppage)->zone_device_data;
			mdevice->calloc += 1UL << compound_order(*ppage);
		} else {
			*ppage = mdevice->free_pages;
			mdevice->free_pages = (*ppage)->zone_device_data;
			mdevice->calloc++;
		}
	}
	spin_unlock(&mdevice->lock);

	return true;

err_free:
	kfree(devmem);
err:
	mutex_unlock(&mdevice->devmem_lock);
	return false;
}

static struct page *dmirror_devmem_alloc_page(struct dmirror_device *mdevice,
					      bool is_huge)
{
	struct page *dpage = NULL;
	struct page *rpage;

	/*
	 * This is a fake device so we alloc real system memory to store
	 * our device memory.
	 */
	rpage = alloc_page(GFP_HIGHUSER);
	if (!rpage)
		return NULL;

	spin_lock(&mdevice->lock);

	if (is_huge && mdevice->free_huge_pages) {
		dpage = mdevice->free_huge_pages;
		mdevice->free_huge_pages = dpage->zone_device_data;
		mdevice->calloc += 1UL << compound_order(dpage);
		spin_unlock(&mdevice->lock);
	} else if (!is_huge && mdevice->free_pages) {
		dpage = mdevice->free_pages;
		mdevice->free_pages = dpage->zone_device_data;
		mdevice->calloc++;
		spin_unlock(&mdevice->lock);
	} else {
		spin_unlock(&mdevice->lock);
		if (!dmirror_allocate_chunk(mdevice, is_huge, &dpage))
			goto error;
	}

	if (is_huge) {
		unsigned int nr_pages = 1U << compound_order(dpage);
		unsigned int i;
		struct page **tpage;

		tpage = kmap(rpage);
		for (i = 0; i < nr_pages; i++, tpage++) {
			*tpage = alloc_page(GFP_HIGHUSER);
			if (!*tpage) {
				while (i--)
					__free_page(*--tpage);
				kunmap(rpage);
				goto error;
			}
		}
		kunmap(rpage);
	}

	dpage->zone_device_data = rpage;
	get_page(dpage);
	lock_page(dpage);
	return dpage;

error:
	__free_page(rpage);
	return NULL;
}

static void dmirror_migrate_alloc_and_copy(struct migrate_vma *args,
					   struct dmirror *dmirror)
{
	struct dmirror_device *mdevice = dmirror->mdevice;
	const unsigned long *src = args->src;
	unsigned long *dst = args->dst;
	unsigned long end_pfn = args->end >> PAGE_SHIFT;
	unsigned long pfn;

	for (pfn = args->start >> PAGE_SHIFT; pfn < end_pfn; ) {
		struct page *spage;
		struct page *dpage;
		struct page *rpage;
		bool is_huge;

		if (!(*src & MIGRATE_PFN_MIGRATE))
			goto next;

		/*
		 * Note that spage might be NULL which is OK since it is an
		 * unallocated pte_none() or read-only zero page.
		 */
		spage = migrate_pfn_to_page(*src);

		/*
		 * Don't migrate device private pages from our own driver or
		 * others. For our own we would do a device private memory copy
		 * not a migration and for others, we would need to fault the
		 * other device's page into system memory first.
		 */
		if (spage && is_zone_device_page(spage))
			goto next;

		/* This flag is only set if a whole huge page is migrated. */
		is_huge = *src & MIGRATE_PFN_HUGE;
		dpage = dmirror_devmem_alloc_page(mdevice, is_huge);
		if (!dpage)
			goto next;

		/*
		 * Normally, a device would use the page->zone_device_data to
		 * point to the mirror but here we use it to hold the page for
		 * the simulated device memory and that page holds the pointer
		 * to the mirror.
		 */
		rpage = dpage->zone_device_data;
		rpage->zone_device_data = dmirror;

		*dst = migrate_pfn(page_to_pfn(dpage)) |
			    MIGRATE_PFN_LOCKED;
		if ((*src & MIGRATE_PFN_WRITE) ||
		    (!spage && args->vma->vm_flags & VM_WRITE))
			*dst |= MIGRATE_PFN_WRITE;

		if (is_huge) {
			struct page **tpage;
			unsigned int order = compound_order(dpage);
			unsigned long endp = pfn + (1UL << order);

			*dst |= MIGRATE_PFN_HUGE;
			tpage = kmap(rpage);
			while (pfn < endp) {
				if (spage) {
					copy_highpage(*tpage, spage);
					spage++;
				} else
					clear_highpage(*tpage);
				tpage++;
				pfn++;
				src++;
				dst++;
			}
			kunmap(rpage);
			continue;
		}

		if (spage)
			copy_highpage(rpage, spage);
		else
			clear_highpage(rpage);
	next:
		pfn++;
		src++;
		dst++;
	}
}

static int dmirror_migrate_finalize_and_map(struct migrate_vma *args,
					    struct dmirror *dmirror)
{
	unsigned long start = args->start;
	unsigned long end = args->end;
	const unsigned long *src = args->src;
	const unsigned long *dst = args->dst;
	unsigned long pfn;
	int ret = 0;

	/* Map the migrated pages into the device's page tables. */
	mutex_lock(&dmirror->mutex);

	for (pfn = start >> PAGE_SHIFT; pfn < (end >> PAGE_SHIFT); ) {
		unsigned long mpfn;
		struct page *dpage;
		struct page *rpage;
		void *entry;

		if (!(*src & MIGRATE_PFN_MIGRATE))
			goto next;

		mpfn = *dst;
		dpage = migrate_pfn_to_page(mpfn);
		if (!dpage)
			goto next;

		/*
		 * Store the page that holds the data so the page table
		 * doesn't have to deal with ZONE_DEVICE private pages.
		 */
		rpage = dpage->zone_device_data;
		if (mpfn & MIGRATE_PFN_HUGE) {
			struct page **tpage;
			unsigned int order = compound_order(dpage);
			unsigned long end_pfn = pfn + (1UL << order);

			ret = 0;
			tpage = kmap(rpage);
			while (pfn < end_pfn) {
				entry = *tpage;
				if (mpfn & MIGRATE_PFN_WRITE)
					entry = xa_tag_pointer(entry,
							DPT_XA_TAG_WRITE);
				entry = xa_store(&dmirror->pt, pfn, entry,
						 GFP_KERNEL);
				if (xa_is_err(entry)) {
					ret = xa_err(entry);
					break;
				}
				tpage++;
				pfn++;
				src++;
				dst++;
			}
			kunmap(rpage);
			if (ret)
				goto err;
			continue;
		}

		entry = rpage;
		if (mpfn & MIGRATE_PFN_WRITE)
			entry = xa_tag_pointer(entry, DPT_XA_TAG_WRITE);
		entry = xa_store(&dmirror->pt, pfn, entry, GFP_ATOMIC);
		if (xa_is_err(entry)) {
			ret = xa_err(entry);
			goto err;
		}
	next:
		pfn++;
		src++;
		dst++;
	}

err:
	mutex_unlock(&dmirror->mutex);
	return ret;
}

static int dmirror_migrate(struct dmirror *dmirror,
			   struct hmm_dmirror_cmd *cmd)
{
	unsigned long start, end, addr;
	unsigned long size = cmd->npages << PAGE_SHIFT;
	struct mm_struct *mm = dmirror->mm;
	struct vm_area_struct *vma;
	unsigned long *src_pfns;
	unsigned long *dst_pfns;
	struct dmirror_bounce bounce;
	struct migrate_vma args;
	unsigned long next;
	int ret;

	start = cmd->addr;
	end = start + size;
	if (end < start)
		return -EINVAL;

	/* Since the mm is for the mirrored process, get a reference first. */
	if (!mmget_not_zero(mm))
		return -EINVAL;

	src_pfns = kmalloc_array(PTRS_PER_PTE, sizeof(*src_pfns), GFP_KERNEL);
	if (!src_pfns) {
		ret = -ENOMEM;
		goto out_put;
	}
	dst_pfns = kmalloc_array(PTRS_PER_PTE, sizeof(*dst_pfns), GFP_KERNEL);
	if (!dst_pfns) {
		ret = -ENOMEM;
		goto out_free_src;
	}

	down_read(&mm->mmap_sem);
	for (addr = start; addr < end; addr = next) {
		vma = find_vma(mm, addr);
		if (!vma || addr < vma->vm_start) {
			ret = -EINVAL;
			goto out;
		}
		next = min(end, addr + (PTRS_PER_PTE << PAGE_SHIFT));
		if (next > vma->vm_end)
			next = vma->vm_end;

		args.vma = vma;
		args.src = src_pfns;
		args.dst = dst_pfns;
		args.start = addr;
		args.end = next;
		ret = migrate_vma_setup(&args);
		if (ret)
			goto out;

		dmirror_migrate_alloc_and_copy(&args, dmirror);
		migrate_vma_pages(&args);
		dmirror_migrate_finalize_and_map(&args, dmirror);
		migrate_vma_finalize(&args);
	}
	kfree(dst_pfns);
	kfree(src_pfns);
	up_read(&mm->mmap_sem);
	mmput(mm);

	/* Return the migrated data for verification. */
	ret = dmirror_bounce_init(&bounce, start, size);
	if (ret)
		return ret;
	mutex_lock(&dmirror->mutex);
	ret = dmirror_do_read(dmirror, start, end, &bounce);
	mutex_unlock(&dmirror->mutex);
	if (ret == 0)
		ret = copy_to_user((void __user *)cmd->ptr, bounce.ptr,
					bounce.size);
	cmd->cpages = bounce.cpages;
	dmirror_bounce_fini(&bounce);
	return ret;

out:
	up_read(&mm->mmap_sem);
	kfree(dst_pfns);
out_free_src:
	kfree(src_pfns);
out_put:
	mmput(mm);
	return ret;
}

static void dmirror_mkentry(struct dmirror *dmirror, struct hmm_range *range,
			    unsigned char *perm, uint64_t entry)
{
	struct page *page;

	if (entry == range->values[HMM_PFN_ERROR]) {
		*perm = HMM_DMIRROR_PROT_ERROR;
		return;
	}
	page = hmm_device_entry_to_page(range, entry);
	if (!page) {
		*perm = HMM_DMIRROR_PROT_NONE;
		return;
	}
	if (entry & range->flags[HMM_PFN_DEVICE_PRIVATE]) {
		/* Is the page migrated to this device or some other? */
		if (dmirror->mdevice == dmirror_page_to_device(page))
			*perm = HMM_DMIRROR_PROT_DEV_PRIVATE_LOCAL;
		else
			*perm = HMM_DMIRROR_PROT_DEV_PRIVATE_REMOTE;
	} else if (is_zero_pfn(page_to_pfn(page)))
		*perm = HMM_DMIRROR_PROT_ZERO;
	else
		*perm = HMM_DMIRROR_PROT_NONE;
	if (entry & range->flags[HMM_PFN_WRITE])
		*perm |= HMM_DMIRROR_PROT_WRITE;
	else
		*perm |= HMM_DMIRROR_PROT_READ;
}

static bool dmirror_snapshot_invalidate(struct mmu_interval_notifier *mni,
				const struct mmu_notifier_range *range,
				unsigned long cur_seq)
{
	struct dmirror_interval *dmi =
		container_of(mni, struct dmirror_interval, notifier);
	struct dmirror *dmirror = dmi->dmirror;

	if (mmu_notifier_range_blockable(range))
		mutex_lock(&dmirror->mutex);
	else if (!mutex_trylock(&dmirror->mutex))
		return false;

	/*
	 * Snapshots only need to set the sequence number since the
	 * invalidations are handled by the dmirror_interval ranges.
	 */
	mmu_interval_set_seq(mni, cur_seq);

	mutex_unlock(&dmirror->mutex);
	return true;
}

static const struct mmu_interval_notifier_ops dmirror_mrn_ops = {
	.invalidate = dmirror_snapshot_invalidate,
};

static int dmirror_range_snapshot(struct dmirror *dmirror,
				  struct hmm_range *range,
				  unsigned char *perm)
{
	struct mm_struct *mm = dmirror->mm;
	struct dmirror_interval notifier;
	unsigned long timeout =
		jiffies + msecs_to_jiffies(HMM_RANGE_DEFAULT_TIMEOUT);
	unsigned long i;
	unsigned long n;
	int ret = 0;

	notifier.dmirror = dmirror;
	range->notifier = &notifier.notifier;

	ret = mmu_interval_notifier_insert_safe(range->notifier, mm,
			range->start, range->end - range->start,
			&dmirror_mrn_ops);
	if (ret)
		return ret;

	while (true) {
		long count;

		if (time_after(jiffies, timeout)) {
			ret = -EBUSY;
			goto out;
		}

		range->notifier_seq = mmu_interval_read_begin(range->notifier);

		down_read(&mm->mmap_sem);
		count = hmm_range_fault(range, HMM_FAULT_SNAPSHOT);
		up_read(&mm->mmap_sem);
		if (count <= 0) {
			if (count == 0 || count == -EBUSY)
				continue;
			ret = count;
			goto out;
		}

		mutex_lock(&dmirror->mutex);
		if (mmu_interval_read_retry(range->notifier,
					    range->notifier_seq)) {
			mutex_unlock(&dmirror->mutex);
			continue;
		}
		break;
	}

	n = (range->end - range->start) >> PAGE_SHIFT;
	for (i = 0; i < n; i++)
		dmirror_mkentry(dmirror, range, perm + i, range->pfns[i]);

	mutex_unlock(&dmirror->mutex);
out:
	mmu_interval_notifier_remove(range->notifier);
	return ret;
}

static int dmirror_snapshot(struct dmirror *dmirror,
			    struct hmm_dmirror_cmd *cmd)
{
	struct mm_struct *mm = dmirror->mm;
	unsigned long start, end;
	unsigned long size = cmd->npages << PAGE_SHIFT;
	unsigned long addr;
	unsigned long next;
	uint64_t pfns[64];
	unsigned char perm[64];
	char __user *uptr;
	struct hmm_range range = {
		.pfns = pfns,
		.flags = dmirror_hmm_flags,
		.values = dmirror_hmm_values,
		.pfn_shift = DPT_SHIFT,
		.pfn_flags_mask = ~0ULL,
	};
	int ret = 0;

	start = cmd->addr;
	end = start + size;
	if (end < start)
		return -EINVAL;

	/* Since the mm is for the mirrored process, get a reference first. */
	if (!mmget_not_zero(mm))
		return -EINVAL;

	/*
	 * Register a temporary notifier to detect invalidations even if it
	 * overlaps with other mmu_interval_notifiers.
	 */
	uptr = (void __user *)cmd->ptr;
	for (addr = start; addr < end; addr = next) {
		unsigned long n;

		next = min(addr + (ARRAY_SIZE(pfns) << PAGE_SHIFT), end);
		range.start = addr;
		range.end = next;

		ret = dmirror_range_snapshot(dmirror, &range, perm);
		if (ret)
			break;

		n = (range.end - range.start) >> PAGE_SHIFT;
		ret = copy_to_user(uptr, perm, n);
		if (ret)
			break;

		cmd->cpages += n;
		uptr += n;
	}
	mmput(mm);

	return ret;
}

static long dmirror_fops_unlocked_ioctl(struct file *filp,
					unsigned int command,
					unsigned long arg)
{
	void __user *uarg = (void __user *)arg;
	struct hmm_dmirror_cmd cmd;
	struct dmirror *dmirror;
	int ret;

	dmirror = filp->private_data;
	if (!dmirror)
		return -EINVAL;

	ret = copy_from_user(&cmd, uarg, sizeof(cmd));
	if (ret)
		return ret;

	if (cmd.addr & ~PAGE_MASK)
		return -EINVAL;
	if (cmd.addr >= (cmd.addr + (cmd.npages << PAGE_SHIFT)))
		return -EINVAL;

	cmd.cpages = 0;
	cmd.faults = 0;

	switch (command) {
	case HMM_DMIRROR_READ:
		ret = dmirror_read(dmirror, &cmd);
		break;

	case HMM_DMIRROR_WRITE:
		ret = dmirror_write(dmirror, &cmd);
		break;

	case HMM_DMIRROR_MIGRATE:
		ret = dmirror_migrate(dmirror, &cmd);
		break;

	case HMM_DMIRROR_SNAPSHOT:
		ret = dmirror_snapshot(dmirror, &cmd);
		break;

	default:
		return -EINVAL;
	}
	if (ret)
		return ret;

	return copy_to_user(uarg, &cmd, sizeof(cmd));
}

static const struct file_operations dmirror_fops = {
	.open		= dmirror_fops_open,
	.release	= dmirror_fops_release,
	.unlocked_ioctl = dmirror_fops_unlocked_ioctl,
	.llseek		= default_llseek,
	.owner		= THIS_MODULE,
};

static void dmirror_devmem_free(struct page *page)
{
	struct page *rpage = compound_head(page)->zone_device_data;
	unsigned int order = compound_order(page);
	unsigned int nr_pages = 1U << order;
	struct dmirror_device *mdevice;

	if (rpage) {
		if (order) {
			unsigned int i;
			struct page **tpage;
			void *kaddr;

			kaddr = kmap_atomic(rpage);
			tpage = kaddr;
			for (i = 0; i < nr_pages; i++, tpage++)
				__free_page(*tpage);
			kunmap_atomic(kaddr);
		}
		__free_page(rpage);
	}

	mdevice = dmirror_page_to_device(page);

	spin_lock(&mdevice->lock);
	if (order) {
		page->zone_device_data = mdevice->free_huge_pages;
		mdevice->free_huge_pages = page;
	} else {
		page->zone_device_data = mdevice->free_pages;
		mdevice->free_pages = page;
	}
	mdevice->cfree += nr_pages;
	spin_unlock(&mdevice->lock);
}

static vm_fault_t dmirror_devmem_fault_alloc_and_copy(struct migrate_vma *args,
						struct dmirror_device *mdevice)
{
	struct vm_area_struct *vma = args->vma;
	const unsigned long *src = args->src;
	unsigned long *dst = args->dst;
	unsigned long start = args->start;
	unsigned long end = args->end;
	unsigned long addr;

	for (addr = start; addr < end; ) {
		struct page *spage, *dpage;
		unsigned int order = 0;
		unsigned int nr_pages = 1;
		unsigned int i;

		spage = migrate_pfn_to_page(*src);
		if (!spage || !(*src & MIGRATE_PFN_MIGRATE))
			goto next;
		order = compound_order(spage);
		nr_pages = 1U << order;
		/* The source page is the ZONE_DEVICE private page. */
		spage = spage->zone_device_data;

		if (order)
			dpage = alloc_transhugepage(vma, addr);
		else
			dpage = alloc_pages_vma(GFP_HIGHUSER_MOVABLE, 0, vma, addr,
						numa_node_id(), false);

		if (!dpage || compound_order(dpage) != order)
			return VM_FAULT_OOM;

		lock_page(dpage);
		*dst = migrate_pfn(page_to_pfn(dpage)) | MIGRATE_PFN_LOCKED;
		if (*src & MIGRATE_PFN_WRITE)
			*dst |= MIGRATE_PFN_WRITE;
		if (order) {
			struct page **tpage;

			*dst |= MIGRATE_PFN_HUGE;
			tpage = kmap(spage);
			for (i = 0; i < nr_pages; i++) {
				copy_highpage(dpage, *tpage);
				tpage++;
				dpage++;
			}
			kunmap(spage);
		} else
			copy_highpage(dpage, spage);
	next:
		addr += PAGE_SIZE << order;
		src += nr_pages;
		dst += nr_pages;
	}
	return 0;
}

static void dmirror_devmem_fault_finalize_and_map(struct migrate_vma *args,
						  struct dmirror *dmirror)
{
	/* Invalidate the device's page table mapping. */
	mutex_lock(&dmirror->mutex);
	dmirror_do_update(dmirror, args->start, args->end);
	mutex_unlock(&dmirror->mutex);
}

static vm_fault_t dmirror_devmem_fault(struct vm_fault *vmf)
{
	struct migrate_vma args;
	unsigned long src_pfns;
	unsigned long dst_pfns;
	struct page *page;
	struct page *rpage;
	unsigned int order;
	struct dmirror *dmirror;
	vm_fault_t ret;

	page = compound_head(vmf->page);
	order = compound_order(page);

	/*
	 * Normally, a device would use the page->zone_device_data to point to
	 * the mirror but here we use it to hold the page for the simulated
	 * device memory and that page holds the pointer to the mirror.
	 */
	rpage = page->zone_device_data;
	dmirror = rpage->zone_device_data;

	if (order) {
		args.start = vmf->address & (PAGE_MASK << order);
		args.end = args.start + (PAGE_SIZE << order);
		args.src = kzalloc(sizeof(*args.src) * PTRS_PER_PTE,
				   GFP_KERNEL);
		if (!args.src)
			return VM_FAULT_OOM;
		args.dst = kzalloc(sizeof(*args.dst) * PTRS_PER_PTE,
				   GFP_KERNEL);
		if (!args.dst) {
			ret = VM_FAULT_OOM;
			goto error_src;
		}
	} else {
		args.start = vmf->address;
		args.end = args.start + PAGE_SIZE;
		args.src = &src_pfns;
		args.dst = &dst_pfns;
	}
	args.vma = vmf->vma;

	if (migrate_vma_setup(&args)) {
		ret = VM_FAULT_SIGBUS;
		goto error_dst;
	}

	ret = dmirror_devmem_fault_alloc_and_copy(&args, dmirror->mdevice);
	if (ret)
		goto error_fin;
	migrate_vma_pages(&args);
	dmirror_devmem_fault_finalize_and_map(&args, dmirror);
	migrate_vma_finalize(&args);
	if (order) {
		kfree(args.dst);
		kfree(args.src);
	}
	return 0;

error_fin:
	migrate_vma_finalize(&args);
error_dst:
	kfree(args.dst);
error_src:
	kfree(args.src);
	return ret;
}

static const struct dev_pagemap_ops dmirror_devmem_ops = {
	.page_free	= dmirror_devmem_free,
	.migrate_to_ram	= dmirror_devmem_fault,
};

static int dmirror_device_init(struct dmirror_device *mdevice, int id)
{
	dev_t dev;
	int ret;

	dev = MKDEV(MAJOR(dmirror_dev), id);
	mutex_init(&mdevice->devmem_lock);
	spin_lock_init(&mdevice->lock);

	cdev_init(&mdevice->cdevice, &dmirror_fops);
	ret = cdev_add(&mdevice->cdevice, dev, 1);
	if (ret)
		return ret;

	/* Build a list of free ZONE_DEVICE private struct pages */
	dmirror_allocate_chunk(mdevice, false, NULL);

	return 0;
}

static void dmirror_device_remove(struct dmirror_device *mdevice)
{
	unsigned int i;

	if (mdevice->devmem_chunks) {
		for (i = 0; i < mdevice->devmem_count; i++) {
			struct dmirror_chunk *devmem =
				mdevice->devmem_chunks[i];

			memunmap_pages(&devmem->pagemap);
			kfree(devmem);
		}
		kfree(mdevice->devmem_chunks);
	}

	cdev_del(&mdevice->cdevice);
}

static int __init hmm_dmirror_init(void)
{
	int ret;
	int id;

	ret = alloc_chrdev_region(&dmirror_dev, 0, DMIRROR_NDEVICES,
				  "HMM_DMIRROR");
	if (ret)
		goto err_unreg;

	for (id = 0; id < DMIRROR_NDEVICES; id++) {
		ret = dmirror_device_init(dmirror_devices + id, id);
		if (ret)
			goto err_chrdev;
	}

	/*
	 * Allocate a zero page to simulate a reserved page of device private
	 * memory which is always zero. The zero_pfn page isn't used just to
	 * make the code here simpler (i.e., we need a struct page for it).
	 */
	dmirror_zero_page = alloc_page(GFP_HIGHUSER | __GFP_ZERO);
	if (!dmirror_zero_page)
		goto err_chrdev;

	pr_info("HMM test module loaded. This is only for testing HMM.\n");
	return 0;

err_chrdev:
	while (--id >= 0)
		dmirror_device_remove(dmirror_devices + id);
	unregister_chrdev_region(dmirror_dev, DMIRROR_NDEVICES);
err_unreg:
	return ret;
}

static void __exit hmm_dmirror_exit(void)
{
	int id;

	if (dmirror_zero_page)
		__free_page(dmirror_zero_page);
	for (id = 0; id < DMIRROR_NDEVICES; id++)
		dmirror_device_remove(dmirror_devices + id);
	unregister_chrdev_region(dmirror_dev, DMIRROR_NDEVICES);
}

module_init(hmm_dmirror_init);
module_exit(hmm_dmirror_exit);
MODULE_LICENSE("GPL");
