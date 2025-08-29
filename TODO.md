# KG Engine TODO

## Known Issues

### Progress Bar Suppression (High Priority)
**Problem**: SentenceTransformer library outputs verbose progress bars ("Batches: 100%|████...") to stderr during model operations, even after setting `show_progress_bar=False` on encode() calls.

**Current Workaround**: Redirecting stderr to /dev/null in Makefile for all kg commands (`2>/dev/null`). This suppresses ALL stderr output, not just progress bars.

**Root Cause**: 
- Progress bars appear during model lazy loading after search results are displayed
- The "Use pytorch device_name: mps" and "Load pretrained SentenceTransformer:" messages indicate model is being loaded multiple times or lazily
- Setting TQDM_DISABLE=1 and transformers.logging.disable_progress_bar() doesn't fully suppress these

**Proper Solution Needed**:
1. Investigate why model is being loaded multiple times (appears to happen during connectivity scoring phase)
2. Implement selective stderr filtering to suppress only tqdm/progress output while preserving actual errors
3. Consider using a context manager to temporarily redirect stderr during specific operations
4. Look into SentenceTransformer's internal batch processing to disable progress at a lower level
5. Possibly cache the model instance to prevent multiple initializations

**Impact**: Users lose visibility of actual errors since all stderr is suppressed. This could make debugging difficult.

## Future Improvements

### Architecture
- [ ] Implement proper model caching to prevent multiple initializations
- [ ] Add --quiet flag for silent operation vs current --debug flag
- [ ] Separate progress/status messages from actual errors

### Testing
- [ ] Add unit tests for all core functions
- [ ] Test with different embedding models
- [ ] Performance benchmarks for large knowledge graphs