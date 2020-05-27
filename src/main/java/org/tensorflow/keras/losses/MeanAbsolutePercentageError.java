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
 * Computes the mean absolute percentage error between labels and predictions.
 * 
 * @author Jim Clarke
 */
public class MeanAbsolutePercentageError extends Loss {
    
    /**
     *  Creates a MeanAbsolutePercentageError with Reduction.AUTO
     */
    public MeanAbsolutePercentageError(Ops tf) {
        super(tf, "mean_squared_error");
    }
    
    /**
     *  Creates a MeanAbsolutePercentageError
     * 
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanAbsolutePercentageError(Ops tf, Reduction reduction) {
        super(tf, "mean_squared_error", reduction);
    }
    
    /**
     *  Creates a MeanAbsolutePercentageError with Reduction.AUTO
     * @param name the name for the loss function
     */
    public MeanAbsolutePercentageError(Ops tf, String name) {
        super(tf, name);
    }
    
    /**
     *  Creates a MeanAbsolutePercentageError
     * 
     * @param name the name for the loss function
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanAbsolutePercentageError(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }
    

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.mean_absolute_percentage_error(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
