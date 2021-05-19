.PHONY: all clean
all:
	$(info "This Makefile has no default target. Try 'make clean'.")

clean:
	${RM} *.pkl.gz
	${RM} *.log
	${RM} *.pkl
