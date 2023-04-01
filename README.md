# PFR-MWM
Annual project at Telecom Paris with MWM. The goal is to develop a framework that creates a music video automatically given video shots and a music.

\section{Introduction}

This report aims to summarize our work done within the context of the fil rouge project in the BGD/IA specialized Master's degree program. The project was proposed by Music World Media (MWM), a company specialized in developing mobile and tablet applications, as well as connected objects in the field of music. The goal of the project is to develop the algorithmic part of an automatic video editing application.

MWM needs an algorithm to create an automatic video montage from video clips and music. The objective is to minimize the user's interactions with the application. The value proposition for this project is to "provide the user with a professional quality montage in one click". To achieve this, MWM wants us to use machine learning models.

This project has three main axes:
\begin{itemize}
\item Music axis: Processing audio to extract useful information for editing
\item Video axis: Establishing an interest score on video segments to rank the best segments for editing
\item Editing axis: Using information from the previous two axes to create a professional quality montage.
\end{itemize}
\bigskip
There is also a bonus fourth axis: optimization for mobile devices. As the application is intended for mobile devices, the algorithms we develop should ideally run on mobile devices. MWM has specified that this objective is a bonus and that the priority is to obtain efficient algorithms without taking into account the execution platform.

We can also add that we did not start from scratch. Indeed, this is the second year that MWM has proposed this topic, so we had access to the previous group's report. We used it to see their approach and the paths they explored. However, it was decided in agreement with MWM not to reuse their work in order to have a new approach to the project.
