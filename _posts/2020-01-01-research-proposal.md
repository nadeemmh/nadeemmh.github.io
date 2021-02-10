---
title: "Research Proposal: Pattern Visualization in
Continuous Person Monitoring Applications"
categories:
  - university
tags:
  - data mining
  - computer science
  - weka
  - data analysis
---

## Introduction
The incorporation of machine learning into clinical medicine holds promise for substantially
improving health care delivery by pairing technology with the overwhelming amounts of
data being transmitted on a daily basis. Enabling data to be accessed remotely, in ways such
as blood pressure monitoring devices for patients, has allowed research to accelerate, and the
large data sets allow for more consistent and credible evaluations.

The techniques used for this project focus heavily on machine learning and how large volumes
of data can be processed to learn complex patterns about individual patients as well
as the conditions they suffer from. These machine learning algorithms are used to determine
characteristics such as classification and clustering. In addition to this, data-based estimation
methods in network science, such as recurrent neural networks will be implemented, as well
as components of probabilistic modelling methods.

The primary aim for applying the machine learning algorithms that will be used in this
study is to support the physical well-being of patients with conditions such as dementia,
Parkinson’s disease, Alzheimer’s, etc. By analysing raw observations and measurement data
alongside environmental data, collected from hospitals and portable person monitoring devices,
the data can be condensed and sorted in a manner where patterns can be found in the
behaviour of the patients and the progress of their underlying conditions.

## 2 Aims and Objectives
The overall aim of this study is to find a novel approach using data-driven techniques for
pattern visualization in continuous person monitoring applications. This means developing
new and innovative techniques for data mining and finding patterns in data collected through
human monitoring. This will be done through the development of creative machine learning
ideas and techniques.

The specific objectives are:

-To identify new data-driven techniques for analysing patterns that are generated in
health applications through continuous person monitoring.

- Using recent advances in data-driven statistical machine learning models, find a new
ways to adapt to constraints and drawbacks.

- Analysing the data and patterns to gain a better understanding of what observations to
focus on.

- To acknowledge applications of the research and adjust to the needs of participants accordingly.

- Understanding the patterns in data and being able to convey these results to clinicians
and/or other medical professionals so that they can provide better care for patients.

In a project of this calibre, there are important questions that must be asked, such as:

- What are all the variables that will be measured to achieve the study outcomes? How
will these variables be measured?

- What are the limitations of the methods, and what can be done to minimize these limitations?

## 3 On-Going Research and Methodology
A preliminary literature review shows that past studies are primarily focused on understanding
and modeling the data collected from patients with various conditions, and finding patterns
and discovering changes in participant’s health and well-being. Big data has helped
transform medicine, which has ultimately advanced the research quality that leads to medical
breakthroughs. That being said, data must be analyzed, interpreted, and acted on for it to
be of any use which is why algorithms have proven to be groundbreaking. As health conditions
become more complex and unusual, clinical medicine will be challenged to grow and
innovate. Thus, machine learning is becoming an indispensable tool for clinicians who seek to
truly understand not only these new conditions, but more importantly their patients.

Machine learning is currently being used in a variety of ways in clinical medicine. A study
by Shirin Enshaeifar, Ahmed Zoha et al. found that environmental and sensory data can be
used to improve the clinical decision making and enhance the care and support given to patients
and their caregivers, Enshaeifar et al. (2018). The study focuses on machine learning
algorithms used to analyse environmental data collected by Internet of Things (IoT) technologies
in Technology Integrated Health Management (TIHM) in order to monitor and assist in
the well being of people with dementia. IoT technologies allow for data to be transferred over
a network without requiring human-to-human interaction. The research implements the use
of ”Markov Chain” which is defined as:

A Markov chain is a stochastic model describing a sequence of possible events
in which the probability of each event depends only on the state attained in the
previous event, Oxford (2020).

This can be used for a stochastic process x = x1, x2, ..., xi􀀀1, xi, where the probability of each
step xi only depends on the value of the preceding step xi􀀀1, and not the entire sequence.

![layers](https://i.imgur.com/e53voTZ.png)

Here, P<sub>i-1,i</sub> is the transition probability of x<sub>i-1 -> i</sub>, Tan and Xi. (2008). This research paper
validates that processes such as the one described above can be used in the proposed project
since there are many similarities in the type of research being conducted. It also shows that
there is a need for this research as it will enable patients who are entirely reliant on their
caregivers to be granted a higher quality of life and care.

Another study conducted by Michael K. K. Leung et al. mentions dramatic gains in accuracy
by moving from Markov models toward deep recurrent models Leung et al. (2015).
The study focuses on using machine learning to tackle key problems in genomic medicine. In
this investigation, hidden Markov models were used in speech recognition due to their ability
of learning better representations of the data. Hidden Markov models (HMMs) are Markov
models in which the system is assumed to be a Markov process with unobservable or hidden
states, Beal et al. (2002). In this project, a possible use of HMMs is to find the relationships between
the different types of data that has been extracted from sensors and identify the routine
activities of the participant. HMM can also be used in time series prediction to predict future
values of the series.

In addition to machine learning methods, it is also important to consider applications in
which these techniques would be useful. The work of Ahmad Lotfi et al. describes an approach
for supporting independent living of the elderly dementia patients by equipping their
home with a simple sensor network to monitor their behaviour, Lotfi et al. (2012). The results
showed that a building equipped with low-level sensors can provide important information
about the status of the occupant. These sensors were used to record the activities of
the occupant, and allowed the caregiver to observe changes in patterns. In addition to these
monitoring sensors, research by Patrick Boissy focuses on reliable automated accident and fall
detection, Boissy et al. (2007). Boissy explains that 60% of nursing home residents encounter
accidents, more specifically falling over, which results in 10% - 15% of these victims sustaining
serious injuries. The study explains that accurate automated fall detection through user-based
motion sensing is an attainable goal, though factors such as injury risk while testing were a
major drawback for obtaining more conclusive results.

Smart buildings can either monitor and collect the information of the user’s activity through
sensor data, or it can communicate with the environment and allow the user to have control.
The former approach is more commonly used for monitoring and behaviour diagnosis, Medjahed
et al. (2009), whereas the latter approach is used as a form of intervention for the user
for the prevention of accidents, and assisting with reminders and memory recollection. A variety
of statistical methods are also used to monitor the activities of occupants in an intelligent
environment. The work of Dante I. Tapia shows the use of Naive Bayesian classifiers which
classify and detect activities using a “tape-on and forget” sensor system, Tapia et al. (2010).
Naive Bayesian classifiers, one of the simplest Bayesian network models, are probabilistic
classifiers that make strong assumptions between features McCallum (2020).

To summarize, this research shows significant progress in using machine learning and data
analysis tools in order to adapt to the different obstacles that various illnesses bring to medicine.
In addition to this, these projects have not only brought significant improvement to their respective
fields, but have also built the foundation on which the further investigations can be
conducted. This is important to note because research is always ongoing and can be adapted
to new ideas and hypotheses to observe different angles, which is what the proposed research
is set out to accomplish.

## 4 Constraints and Drawbacks
Every machine learning project which uses human data is unique and has its own sets of technical
requirements. As a result, the analysis of this data is subject to numerous constraints that
limit the methodology, which invariably have a significant impact on overall project and research
performance. These challenges include determining the accuracy of different machine
learning techniques and implementing the most suited method for this study. In addition to
this, since data will be explained through charts, graphs and other visualization techniques,
as data volume goes up, this method begins to reach its limits, Zhou et al. (2017).

It is very important to consider the ethics for the present study as it will have a significant
impact on the overall research. Some challenges are straightforward and can be countered,
such as concerns that algorithms may mimic human biases in decision making. A study by
Danton Char et al. explains the various ethical constraints and challenges of conducting a
medical experiment. Char explains that in the U.S. health care system, there is a direct clash
between the goals of improving healthcare, and generating profit Char et al. (2018). This is
a major cause for concern as potential differences between the intent behind the design of
machine learning systems, and the goals of its implementer may create ethical strain. To overcome
this challenge, machine learning can be deployed to help resolve these differences in
health care delivery if algorithms are able to compensate for known biases or identify areas
where research is required.

Due to the magnitude of the project, there will be fail-safe procedures put into place that
allow the project to go on should the initial approach result in failure. It extremely important
understand the significance of this as initial approaches often result in failure due to time constraints,
financial bottlenecks and commonly, a dead-end approach. A ”Plan B”, ”Plan C”, etc.
will be predetermined and developed as the project progresses. These backup plans include
changing the softwares being used to better understand the entirety of the problem, approaching
the problem with a different angle, and metaphorically ”zooming out” and focusing away
from the initial problem.

## 5 Concluding Remarks
Conducting this research will allow for advancements in medicine that assist clinicians, as
well as find new and improved technologies to support patients who require constant supervision.
By assessing past research that focuses on continuous person monitoring, it is clear
that research has made significant changes for the safety and comfort of patients with neurological
disorders such as Alzheimer’s that challenge the cognitive abilities of it’s sufferers.
Though there is a lot of research that has proven to be beneficial, there are a lot of challenges
of this study that have not been overcome, such as consistent accident and fall detection technologies
and even fully functioning smart homes that can be left unsupervised. This project
will attempt to tackle many of these issues and will focus on innovative approaches to machine
learning as well as its potential applications. This project intends to act as a catalyst of
good-quality research in the field of machine learning and clinical medicine as a whole.

## References
M. J. Beal, Z. Ghahramani, and C. E. Rasmussen. The infinite hidden markov model. 2002.

P. Boissy, S. Choquette, M. Hamel, and N. Noury. User-based motion sensing and fuzzy logic
for automated fall detection in older adults. TELEMEDICINE AND e-HEALTH, 13:683—-
693, 2007.

D. S. Char, N. H. Shah, and D. Magnus. Implementing machine learning in health care —
addressing ethical challenges. The New England Journal of Medicine., 378:981—-983, 03 2018.
doi: 10.1056/NEJMp1714229.

S. Enshaeifar, A. Zoha, A. Markides, S. Skillman, S. Acton, and T. Elsaleh. Health management
and pattern analysis of daily living activities of people with dementia using in-home sensors
and machine learning techniques. PLoS ONE, 13(5): e0195605, 2018.

M. K. K. Leung, A. Delong, A. Baba, and B. Frey. Machine Learning in Genomic Medicine: A
Review of Computational Problems and Data Sets, volume 104. 2015.

A. Lotfi, C. Langensiepen, S. M. Mahmoud, and M. J. Akhlaghinia. Smart homes for the
elderly dementia sufferers. Journal of Ambient Intelligence and Humanized Computing, 3:205—
-218, 2012.

A. McCallum. Lecture 2: Bayesian Network Representation. University of Massachusetts
Amherst, Massachusetts , United States of America, 2020.

H. Medjahed, D. Istrate, J. Boudy, and B. Dorizzi. A fuzzy logic system for home elderly people
monitoring (emutem). 03 2009.

Oxford. Markov chain—Definition of Markov chain in US English by Oxford Dictionaries. Oxford
Dictionaries, Oxford, United Kingdom, 2020.

X. Tan and H. Xi. Hidden semi-markov model for anomaly detection. Applied Mathematics and
Computation, 205:562–567, 2008.

D. Tapia, A. Abraham, J. Corchado Rodr´ıguez, and R. Alonso. Agents and ambient intelligence:
Case studies. J. Ambient Intelligence and Humanized Computing, 1:85–93, 06 2010. doi:
10.1007/s12652-009-0006-2.

L. Zhou, S. Pan, J. Wang, and A. V. Vasilakos. Machine learning on big data: Opportunities
and challenges. Neurocomputing, 237:350–361, 2017.
