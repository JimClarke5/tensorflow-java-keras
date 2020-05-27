/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Computes the mean of absolute difference between labels and predictions.
 * @author Jim Clarke
 */
public class MeanAbsoluteError extends Loss {
    /**
     * Creates a MeanAbsoluteError with Reduction.AUTO
     */
    public MeanAbsoluteError(Ops tf) {
        super(tf, "mean_absolute_error");
    }
    
    /**
     * Creates a MeanAbsoluteError
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanAbsoluteError(Ops tf, Reduction reduction) {
        super(tf, "mean_absolute_error", reduction);
    }
    
    /**
     * Creates a MeanAbsoluteError
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanAbsoluteError(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.mean_absolute_error(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
