from pytldr.summarize import TextRankSummarizer
from pytldr.nlp import Tokenizer

tokenizer = Tokenizer('english')
summarizer = TextRankSummarizer(tokenizer)

# If you don't specify a tokenizer when intiializing a summarizer then the
# English tokenizer will be used by default
summarizer = TextRankSummarizer()  # English tokenizer used

# This object creates a summary using the summarize method:
# e.g. summarizer.summarize(text, length=5, weighting='frequency', norm=None)

# The length parameter specifies the length of the summary, either as a
# number of sentences, or a percentage of the original text

# The summarizer can take as input...
# 1. A string:
summary = summarizer.summarize("Some long article bla bla...", length=4)

# 2. A text file:
summary = summarizer.summarize("/path/to/file.txt", length=0.25)
# Above summary is a quarter of the length of the original text

# 3. A URL (must start with http://):
summary = summarizer.summarize("http://newsite.com/some_article")