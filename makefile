SRCFILES=$(wildcard src/posets/*.py)
WHL=dist/posets-$(VERSION)-py3-none-any.whl
PYDOX:=$(shell readlink $$(which pydox))
VERSION:=$(shell grep 'version = ".*"' pyproject.toml -o | rpl -q 'version = "(.*)\.[0-9]+\.[0-9]+\.[0-9]+"' "\1").$(shell date +"%y.%m.%d")
#Why doesn't this regex work?
#VERSION=$(shell cat pyproject.toml | perl -e 'print <STDIN>=~s/version = "(.*)"/$$1/gr').$(shell date +"%y.%m.%d")
NULL:=
SPACE:=$(NULL) #trick from 6.2.2 https://www.gnu.org/software/make/manual/make.html#Flavors
COMMA:=,

$(WHL) : $(SRCFILES) set_version
	hatch build

set_version :
	rpl -q 'version = ".*"' "version = "'"'"$(VERSION)"'"' pyproject.toml

test :
	cd tests; pytest posets_test.py

coverage : tests/htmlcov/index.html

tests/htmlcov/index.html : $(SRCFILES) tests/posets_test.py
	cd tests; coverage run --source ../src/posets -m pytest posets_test.py; coverage html

docs/bib.tex :
	echo '\bibliography{bib}{}' > docs/bib.tex
	echo '\bibliographystyle{plain}' >> docs/bib.tex

docs : $(SRCFILES) docs/posets.sty docs/doc_funcs.py docs/bib.tex $(PYDOX)
	cd docs; pydox ../src/posets -c -I doc_funcs

clean :
	rm -rf docs/figures/*
	rm -rf docs/*.aux docs/*.blg docs/*.log docs/*.toc docs/*.bbl docs/*.out docs/*.toc docs/*.tex
