include ../os.mak

CFLAGS = $(CFLAGS2)
LIBS = $(LIBS2) -lc -lmordb -lmorutil -lport  
OUTFILE = ascbin
OBJECTS = ascbin.o ascii1.o


$(OUTFILE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(OUTFILE) $(LIBS)
	chmod o-rwx $(OUTFILE) 
	cp $(OUTFILE) ../bin
#	strip $(OUTFILE)

revision:
	@ci -l *.[ch]
	@ci -l Makefile


clean: 
	rm -f  $(OBJECTS) core $(OUTFILE) out
