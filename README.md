# Proposals
My ideas for my final project are...

1. Analyze behavioral experiment data and build a model to decompose features from word memorability
  Process experiment data to calculate memorability score of ~400 Chinese word stimulus and tests their consistency.
  Capture multiple feature data from online sources and label word stimulus with different features(word frequency, semantic centrality, sentiment, concreteness, liveliness etc.)
  Use a regression model to predict word memorability based on these features and make statistical inferences about features
  Construct a decision tree model to see whether it can effectivelly predict word memorability

2. Write an automated constructor of memorability-related online experiment programs with various customization options
  Customize the experiment paradigm with various features like frequency and order control of test trials, stimulus display options and participant response modes.
  Construct structred record files packaged in classes, with built-in individual-level and population-level analysis methods
  Experiment record files can be uploaded to an online storage for experimenter to collect
