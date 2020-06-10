### Makefile ---

# targets
.PHONY: all clean lib

all: lib

lib:
	cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
	make -C build
# 	ar cr libsharpSAT.a build/libsharpSAT.a build/component_types/libcomponent_types.a
	cp build/libsharpSAT.a .
	# cp build/component_types/libcomponent_types.a .

clean:
	@-rm $(OBJECTS)

allclean:
	@-find . -name '*.o' -delete
	@-find . -name '*.a' -delete
