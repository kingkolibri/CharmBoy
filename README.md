# CharmBoy No. 1

Welcome to CharmBoy, your parents favorite robot in-law. It talks nice at you and highlights your beautiful characteristics with only one goal, to make you smile.

|  ![](https://github.com/kingkolibri/CharmBoy/blob/master/data/neutral.jpg)  | ![](https://github.com/kingkolibri/CharmBoy/blob/master/data/buddy.jpg)  |  ![](https://github.com/kingkolibri/CharmBoy/blob/master/data/shy.jpg)|   ![](https://github.com/kingkolibri/CharmBoy/blob/master/data/inlove.jpg)|    ![](https://github.com/kingkolibri/CharmBoy/blob/master/data/macho.jpg)|

## What do I need to install?

This module uses opencv and tensorflow. Other packages with the current versions listed in the requirements.txt.

## How can I use it?

Ideally you start the CharmBoy node, the built-in Roboycamera provides the necessary video input and Roboy compliments you on your appearance. 

## What does it actually do underneath? 

The camera input is passed in individual frames to a face classifier, which detects and crops faces. For each identified face, the individual facial features, such as the eye color, are detected in order to select an according compliment from the compliment_database.csv. The inspected features receive a score from 1 to 10, 10 beeing the highest. The feature with the highest score determines the compliment category, e.g. subject has green_eyes with the score of 8 and roboy detects a smile with the score of 4, therefore she will receive a compliment on her green eyes.

## How could I make it better?

The project could be extended upon:
 * Add further features and expand compliment base.
 * Reinforcemnt learning: incorporate feedback from subjects, like blushing, smiling or verbal responsens ("thank you"), to learn how the compliment was received.
 * Setting up a database to store subjects and cross reference them with well-received compliments. 
 * Rate already received compliments with a lower score to highlight each individual treat of the subject.
 * Adding personalities: adapt voice and facial expression to respective compliments for the appropriate occasion.
 * Compare the detected face to an image averaged over all detected faces to tell the subject which of their facial characteristic is especially beautiful.
 * Spread the love <3
