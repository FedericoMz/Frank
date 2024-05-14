# A Frank System for Co-evolutionary Hybrid Decision Making
We introduce Frank, a human-in-the-loop system for co-evolutionary Hybrid Decision-Making. As in traditional HDM systems, a model helps the user labelling records from an un-labelled dataset (or, more generally, making decisions).
Frank employs an incremental learning model to "evolve" in parallel with the user's decisions, by training an interpretable machine learning model on the records labelled by the user, to help them stay consistent over time.

Furthermore, Frank advances the current state-of-the-art approach, namely Skeptical Learning, by checking the user's consistency with the rules given by an external supervisor (Ideal Rule Check) and the user's fairness either w.r.t. the labels assigned to past records (Individual Fairness Check) or proportionally (Group Fairness Check).
We evaluated Frank by simulating the users' behaviour with various levels of expertise and reliance on Frank's suggestions. Our experiments showed that Frank's intervention improved the accuracy and/or the fairness of the decisions. 

![Frank's Steps](https://github.com/FedericoMz/Frank/assets/80719913/4713ade4-57aa-4b1b-b114-a098bf537098)

## What the various files do
_IDA_frank_main_ includes the main systems, whereas _frank_algo_ some auxiliary methods employed by Frank (for example, to compute fairness, or create synthetic records). _classes_ includes various simulated users. Check the tutorial to see how to set the users and Frank.
