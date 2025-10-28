include ../../src/Makefile.include

TARGET := primality
SOURCES := Primality.C
CI := primality.ci

all: $(TARGET)

$(TARGET): $(CI) $(SOURCES)
	$(CHARMC) $(CI)
	$(CHARMC) -o $@ $(SOURCES)

clean:
	-$(RM) $(TARGET) *.decl.h *.def.h *.o charmrun

.PHONY: all clean
