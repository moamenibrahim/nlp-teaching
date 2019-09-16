from pytldr.summarize import LsaSummarizer, LsaOzsoy, LsaSteinberger

summarizer = LsaOzsoy()
summarizer = LsaSteinberger()
summarizer = LsaSummarizer()  # This is identical to the LsaOzsoy object

summary = summarizer.summarize(
    text, topics=4, length=5, binary_matrix=True, topic_sigma_threshold=0.5
)

# topics specifies the number of topics to cluster the article into.
# topic_sigma_threshold removes all topics with a singular value less than a given
# percentage of the largest singular value.
