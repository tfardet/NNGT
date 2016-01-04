======================================
Precise spike timing and interpolation
======================================

Precise spike timing
====================

The implementation with the recovery pseudo event should be kept (cleaner) and the offset needs to be ``step - spike_offset`` because regular (on-grid) spikes should arrive at the end of each step.

Interpolation
=============

Pre-existing implementation (make them more general using the state vector.
