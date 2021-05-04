SOURCES = $(shell find *.py.md)
TARGETS = $(SOURCES:%.py.md=%.html)

default: $(TARGETS)

clean:
	rm $(TARGETS)

%.ipynb: %.py.md
	# See https://jupytext.readthedocs.io/en/latest/using-cli.html
	jupytext --to notebook --output $@ $<

%.html: %.ipynb data.csv
	# https://stackoverflow.com/a/47773056
	# https://nbconvert.readthedocs.io/en/latest/execute_api.html#execution-arguments-traitlets
	time jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 --to html $<

.PRECIOUS: %.ipynb
