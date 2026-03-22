#ifndef PAGE_CACHE_H_
#define PAGE_CACHE_H_

#include <cstring>
#include <cstdint>
#include "ssd_index_defs.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils/lock_table.h"

namespace pipeann {
  // User-space page cache for update acceleration (in fact it's a buffer)
  // only used for write-write, ensure that disk has a consistent state
  // expect a lock-free read

  struct PageCacheItem {
    uint8_t *buf;
    uint64_t ref_cnt;

    // use lock!
    uint64_t ref() {
      return ++ref_cnt;
    }

    // use lock!
    uint64_t deref() {
      return --ref_cnt;
    }
  };

  struct PageCache {
    bool get(uint64_t block_no, uint8_t *value, bool ref = false) {
      bool ret = cache.update_fn(block_no, [&](PageCacheItem &v) {
        memcpy(value, v.buf, SECTOR_LEN);
        if (ref) {
          v.ref();
        }
      });
      return ret;
    }

    bool put(uint64_t block_no, uint8_t *value, bool ref = false) {
      return cache.upsert(block_no, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          v = PageCacheItem{.buf = new uint8_t[SECTOR_LEN], .ref_cnt = 0};
        }
        if (ref) {
          v.ref();
        }
        memcpy(v.buf, value, SECTOR_LEN);
      });
    }

    bool deref(uint64_t block_no) {
      bool ret = cache.uprase_fn(block_no, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          LOG(ERROR) << "PageCache: deref a non-exist block_no: " << block_no;
          return true;
          __builtin_trap();
        }
        uint64_t refs = v.deref();
        if (refs == 0) {
          delete[] v.buf;
        }
        return refs == 0;
      });
      return ret;
    }

    void clear() {
      cache.clear();
    }

    SparseLockTable<uint64_t> lock_table;
    libcuckoo::cuckoohash_map<uint64_t, PageCacheItem> cache;
  };

  inline PageCache cache;
}  // namespace pipeann

#endif  // PAGE_CACHE_H_