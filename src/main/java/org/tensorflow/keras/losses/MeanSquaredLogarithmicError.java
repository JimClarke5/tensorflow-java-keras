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
    public MeanSquaredLogarithmicError() {
        super("mean_squared_logarithmic_error");
    }
    
    /**
     *  Creates a MeanSquaredLogarithmicError
     * 
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanSquaredLogarithmicError(Reduction reduction) {
        super("mean_squared_logarithmic_error", reduction);
    }
    
    /**
     *  Creates a MeanSquaredLogarithmicError with Reduction.AUTO
     */
    public MeanSquaredLogarithmicError(String name) {
        super(name);
    }
    
    /**
     *  Creates a MeanSquaredLogarithmicError
     * 
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanSquaredLogarithmicError(String name, Reduction reduction) {
        super(name, reduction);
    }
    

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Ops tf,  Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
         tf = tf.withSubScope(this.getName());
        Operand losses = Losses.mean_squared_logarithmic_error(tf, labels, predictions);
        return super.computeWeightedLoss(tf, losses, getReduction(), sampleWeights);
    }
    
}
