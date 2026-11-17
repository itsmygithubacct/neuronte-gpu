# The backend chooser: the loader library, the two setup steps and their
# tests. Included by the Makefile above it, or used alone:
#
#   make -f chooser/chooser.mk NEURONTE_DIR=/path/to/neuronte chooser
#   make -f chooser/chooser.mk NEURONTE_DIR=/path/to/neuronte test-chooser
#
# NEURONTE_DIR is a built Neuronte tree: include/neuronte.h and
# build/libneuronte.a (built with -fPIC, as its Makefile does).
# NEURONTE_GPU_INCLUDE is where neuronte_gpu.h is. Nothing here needs a
# GPU, a GPU driver or the GPU library itself: the tests use a stand-in.
CHOOSER_DIR ?= chooser
CHOOSER_BUILD ?= build/chooser
NEURONTE_GPU_INCLUDE ?= src
CHOOSER_NEURONTE_LIB ?= $(NEURONTE_DIR)/build/libneuronte.a
# The smallest net the reference tree ships, for the tests.
CHOOSER_TEST_NRT ?= $(NEURONTE_DIR)/weights/neuronte-braille-v1.nrt
# The runner neuronte-setup, when there is one to test the steps under.
CHOOSER_SETUP ?=

CHOOSER_CC = $(CC) -std=c11 -O2 -g -Wall -Wextra -Wno-unused-function -fPIC \
	-I$(CHOOSER_DIR) -I$(NEURONTE_GPU_INCLUDE) -I$(NEURONTE_DIR)/include $(CPPFLAGS) $(CFLAGS)
CHOOSER_HEADERS = $(CHOOSER_DIR)/common.h $(CHOOSER_DIR)/decide.h $(CHOOSER_DIR)/neuronte_backend.h \
	$(NEURONTE_GPU_INCLUDE)/neuronte_gpu.h

.PHONY: chooser test-chooser install-chooser
chooser: $(CHOOSER_BUILD)/libneuronte-backend.so.0 $(CHOOSER_BUILD)/neuronte-backend-step $(CHOOSER_BUILD)/neuronte-backend

$(CHOOSER_BUILD):
	mkdir -p $(CHOOSER_BUILD)

# The reference library is linked in and kept to itself: the library exports
# the five neuronte_backend_ calls and nothing else.
$(CHOOSER_BUILD)/libneuronte-backend.so.0: $(CHOOSER_DIR)/neuronte_backend.c $(CHOOSER_HEADERS) $(CHOOSER_NEURONTE_LIB) | $(CHOOSER_BUILD)
	$(CHOOSER_CC) -shared -Wl,-soname,libneuronte-backend.so.0 -Wl,--exclude-libs,ALL -o $@ \
		$(CHOOSER_DIR)/neuronte_backend.c $(CHOOSER_NEURONTE_LIB) -lm -ldl -lpthread $(LDFLAGS)
	ln -sf libneuronte-backend.so.0 $(CHOOSER_BUILD)/libneuronte-backend.so

# One program, installed under the names of the two steps.
$(CHOOSER_BUILD)/neuronte-backend-step: $(CHOOSER_DIR)/neuronte-backend-step.c $(CHOOSER_HEADERS) $(CHOOSER_NEURONTE_LIB) | $(CHOOSER_BUILD)
	$(CHOOSER_CC) -o $@ $(CHOOSER_DIR)/neuronte-backend-step.c $(CHOOSER_NEURONTE_LIB) -lm -ldl -lpthread $(LDFLAGS)

# A command that prints the choice and tries a file through the loader.
$(CHOOSER_BUILD)/neuronte-backend: $(CHOOSER_DIR)/neuronte-backend.c $(CHOOSER_BUILD)/libneuronte-backend.so.0 $(CHOOSER_HEADERS)
	$(CHOOSER_CC) -o $@ $(CHOOSER_DIR)/neuronte-backend.c -L$(CHOOSER_BUILD) -lneuronte-backend \
		-Wl,-rpath,'$$ORIGIN' $(CHOOSER_NEURONTE_LIB) -lm

$(CHOOSER_BUILD)/libfake_gpu.so: $(CHOOSER_DIR)/tests/fake_gpu.c $(NEURONTE_GPU_INCLUDE)/neuronte_gpu.h $(CHOOSER_NEURONTE_LIB) | $(CHOOSER_BUILD)
	$(CHOOSER_CC) -shared -Wl,--exclude-libs,ALL -o $@ $(CHOOSER_DIR)/tests/fake_gpu.c $(CHOOSER_NEURONTE_LIB) -lm

$(CHOOSER_BUILD)/test_decide: $(CHOOSER_DIR)/tests/test_decide.c $(CHOOSER_HEADERS) | $(CHOOSER_BUILD)
	$(CHOOSER_CC) -o $@ $(CHOOSER_DIR)/tests/test_decide.c

$(CHOOSER_BUILD)/test_loader: $(CHOOSER_DIR)/tests/test_loader.c $(CHOOSER_BUILD)/libneuronte-backend.so.0 $(CHOOSER_HEADERS)
	$(CHOOSER_CC) -o $@ $(CHOOSER_DIR)/tests/test_loader.c -L$(CHOOSER_BUILD) -lneuronte-backend \
		-Wl,-rpath,'$$ORIGIN' $(CHOOSER_NEURONTE_LIB) -lm

# The loader built into the test together with a stand-in cells file.
$(CHOOSER_BUILD)/test_loader_cells: $(CHOOSER_DIR)/tests/test_loader.c $(CHOOSER_DIR)/tests/fake_cells.c $(CHOOSER_DIR)/neuronte_backend.c $(CHOOSER_HEADERS) $(CHOOSER_NEURONTE_LIB) | $(CHOOSER_BUILD)
	$(CHOOSER_CC) -o $@ $(CHOOSER_DIR)/tests/test_loader.c $(CHOOSER_DIR)/tests/fake_cells.c \
		$(CHOOSER_DIR)/neuronte_backend.c $(CHOOSER_NEURONTE_LIB) -lm -ldl -lpthread

# The loader's cases again with the loader compiled into the test under a sanitizer:
# make test-chooser-sanitized
$(CHOOSER_BUILD)/test_loader_asan: $(CHOOSER_DIR)/tests/test_loader.c $(CHOOSER_DIR)/neuronte_backend.c $(CHOOSER_HEADERS) $(CHOOSER_NEURONTE_LIB) | $(CHOOSER_BUILD)
	$(CHOOSER_CC) -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ $(CHOOSER_DIR)/tests/test_loader.c \
		$(CHOOSER_DIR)/neuronte_backend.c $(CHOOSER_NEURONTE_LIB) -lm -ldl -lpthread
$(CHOOSER_BUILD)/test_loader_tsan: $(CHOOSER_DIR)/tests/test_loader.c $(CHOOSER_DIR)/neuronte_backend.c $(CHOOSER_HEADERS) $(CHOOSER_NEURONTE_LIB) | $(CHOOSER_BUILD)
	$(CHOOSER_CC) -fsanitize=thread -fno-omit-frame-pointer -o $@ $(CHOOSER_DIR)/tests/test_loader.c \
		$(CHOOSER_DIR)/neuronte_backend.c $(CHOOSER_NEURONTE_LIB) -lm -ldl -lpthread
.PHONY: test-chooser-sanitized
test-chooser-sanitized: $(CHOOSER_BUILD)/test_loader_asan $(CHOOSER_BUILD)/test_loader_tsan test-chooser

test-chooser: chooser $(CHOOSER_BUILD)/libfake_gpu.so $(CHOOSER_BUILD)/test_decide $(CHOOSER_BUILD)/test_loader $(CHOOSER_BUILD)/test_loader_cells
	sh $(CHOOSER_DIR)/tests/run.sh $(CHOOSER_BUILD) $(CHOOSER_TEST_NRT) $(CHOOSER_SETUP)

# The loader for programs; the steps for neuronte-setup, in both of its
# directories (40-backend-warm runs for the system and for each user).
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
SETUPDIR ?= $(PREFIX)/lib/neuronte
install-chooser: chooser
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(SETUPDIR)/setup.d $(DESTDIR)$(SETUPDIR)/user-setup.d
	install -m 755 $(CHOOSER_BUILD)/libneuronte-backend.so.0 $(DESTDIR)$(LIBDIR)/
	ln -sf libneuronte-backend.so.0 $(DESTDIR)$(LIBDIR)/libneuronte-backend.so
	install -m 644 $(CHOOSER_DIR)/neuronte_backend.h $(DESTDIR)$(PREFIX)/include/
	install -m 755 $(CHOOSER_BUILD)/neuronte-backend-step $(DESTDIR)$(SETUPDIR)/neuronte-backend-step
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(CHOOSER_BUILD)/neuronte-backend $(DESTDIR)$(PREFIX)/bin/
	ln -sf ../neuronte-backend-step $(DESTDIR)$(SETUPDIR)/setup.d/10-backend
	ln -sf ../neuronte-backend-step $(DESTDIR)$(SETUPDIR)/setup.d/40-backend-warm
	ln -sf ../neuronte-backend-step $(DESTDIR)$(SETUPDIR)/user-setup.d/40-backend-warm
