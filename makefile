.SUFFIXES:
SRCFILES=$(shell git ls-files src/posets/*.py)
DATE:=$(shell date +".%y.%m.%d" | sed -e s/\.0/\./g)
WHL=dist/posets-$(VERSION)$(DATE)-py3-none-any.whl
PYDOX:=$(realpath $(shell which pydox))
VERSION:=$(shell grep -o 'version = "[^"]*' pyproject.toml | sed -e 's/version = "\([^"*]\)/\1/g')
TIMESTAMP:=$(shell date +"%y%m%d%H%M.%S")

#builds whl file for distribution using version from pyproject.toml or commandline if set and appending the current date
#set DATE= on commandline to build without the date in the version
$(WHL) : pyproject.toml $(SRCFILES)
	#set version
	sed -e 's/version = "\([^"*]\)/version = "\1$(DATE)/g' $< > $<. && mv $<. $<
	#build
	hatch build
	#unset version
	sed -e 's/version = "[^"]*"/version = "$(VERSION)"/g' $< > $<. && mv $<. $<
	#fix timestamp for future make calls
	touch -t $(TIMESTAMP) $<

#documention recipes
docs : docs/posets.pdf

docs/bib.tex :
	printf '\\bibliography{bib}{}\n\\bibliographystyle{plain}' > $@

docs/posets.pdf : $(SRCFILES) docs/posets.sty docs/doc_funcs.py docs/bib.tex $(PYDOX)
	cd docs; $(PYDOX) --module ../src/posets --compile bibtex --impall doc_funcs --date today --author William\ Gustafson --title Posets --post bib.tex

#test recipes
test :
	cd tests; pytest posets_test.py -vvv

coverage : tests/htmlcov/index.html

tests/htmlcov/index.html : $(SRCFILES) tests/posets_test.py
	cd tests; coverage run --source ../src/posets -m pytest posets_test.py; coverage html

#removes latex intermediary files and hatch build outputs
clean :
	rm -rf docs/figures/*
	rm -rf docs/*.aux docs/*.blg docs/*.log docs/*.toc docs/*.bbl docs/*.out docs/*.toc docs/*.tex
	rm -rf dist/*
