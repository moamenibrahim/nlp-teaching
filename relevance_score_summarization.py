from pytldr.summarize import RelevanceSummarizer

summarizer = RelevanceSummarizer()
summary = summarizer.summarize(text, length=5, binary_matrix=True):