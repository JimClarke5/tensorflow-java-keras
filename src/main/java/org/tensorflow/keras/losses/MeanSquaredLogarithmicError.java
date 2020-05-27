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
 * Computes the mean squared logarithmic error between labels and predictions.
 * 
 * @author Jim Clarke
 */
public class MeanSquaredLogarithmicError extends Loss {
    
    /**
     *  Creates a MeanSquaredLogarithmicError with Reduction.AUTO
     */
    public MeanSquaredLogarithmicError(Ops tf) {
        super(tf, "mean_squared_logarithmic_error");
    }
    
    /**
     *  Creates a MeanSquaredLogarithmicError
     * 
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanSquaredLogarithmicError(Ops tf, Reduction reduction) {
        super(tf, "mean_squared_logarithmic_error", reduction);
    }
    
    /**
     *  Creates a MeanSquaredLogarithmicError with Reduction.AUTO
     */
    public MeanSquaredLogarithmicError(Ops tf, String name) {
        super(tf, name);
    }
    
    /**
     *  Creates a MeanSquaredLogarithmicError
     * 
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanSquaredLogarithmicError(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }
    

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.mean_squared_logarithmic_error(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
