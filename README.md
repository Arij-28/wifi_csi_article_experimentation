# Reliable Cross-User Wi-Fi CSI Recognition

This repository contains the experimental code and analysis for the paper:

**Reliable Cross-User Wi-Fi CSI Recognition: BiGRU Baselines, Same-Class Mixup, and Reliability-Aware Evaluation**

The project studies raw Wi-Fi Channel State Information (CSI) for human activity recognition under strict cross-user shift. The goal is to evaluate not only classification performance, but also prediction reliability through calibration and selective prediction metrics.

---

## Overview

Wi-Fi CSI enables contactless and privacy-preserving human activity recognition. However, CSI-based models trained on a set of source users may degrade when evaluated on unseen users because CSI patterns depend on user identity, motion style, body shape, orientation, and multipath conditions.

This repository investigates this problem on Widar 3.0 using a strict leave-one-user-out protocol.

The study focuses on:

1. **Temporal backbone strength**  
   Comparing GRU and BiGRU encoders for raw CSI recognition.

2. **Same-class cross-user Mixup**  
   Mixing samples from different source users only when they belong to the same activity class.

3. **Reliability-aware evaluation**  
   Reporting calibration and selective-prediction metrics in addition to Accuracy and Macro-F1.

---

## Main Findings

The main empirical findings are:

- Replacing a GRU with a BiGRU substantially improves cross-user recognition on a representative held-out-user split.
- Same-class cross-user Mixup has a user-dependent effect.
- Mixup improves calibration-related metrics on average, especially ECE and Brier score.
- ERM remains slightly stronger in aggregate Accuracy, Macro-F1, and AURC.
- Cross-user raw-CSI recognition is heterogeneous across held-out users.
- Accuracy alone is insufficient to characterize model behavior under unseen-user shift.

The main interpretation is that same-class cross-user Mixup should be viewed as a calibration-oriented regularizer with user-dependent effects, not as a universally superior classifier.

---

## Dataset

Experiments are conducted on **Widar 3.0**, a Wi-Fi CSI gesture recognition dataset designed around cross-domain variability.

Each processed sample is represented as a normalized CSI tensor:

```text
22 × 400
