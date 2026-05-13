**Project description**
*CI_Artifact Reduction*

Filtering CI Artifacts based on Correlation of Indepndent Components and audio 

*Method Overview*
1. EEG data is decomposed into independent components using ICA.
2. Each component is cross-correlated with the audio stimulus.
3. The peak correlation within an window (can be adjusted) is identified.
4. The SNR value is computed from the correlation peak.
5. Components with SNR values above a defined threshold are excluded.
6. The cleaned EEG signal is reconstructed from the remaining components.
