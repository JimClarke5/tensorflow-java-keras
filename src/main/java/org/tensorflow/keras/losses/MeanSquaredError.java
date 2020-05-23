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
 * Computes the mean of squares of errors between labels and predictions.
 *
 * @author Jim Clarke
 */
public class MeanSquaredError extends Loss {

    /**
     * Creates a MeanSquaredError with Reduction.AUTO
     */
    public MeanSquaredError() {
        super("mean_squared_error");
    }

    /**
     * Creates a MeanSquaredError
     *
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanSquaredError(Reduction reduction) {
        super("mean_squared_error", reduction);
    }
    
     /**
     * Creates a MeanSquaredError with Reduction.AUTO
     * @param name the name for the loss function
     */
    public MeanSquaredError(String name) {
        super(name);
    }

    /**
     * Creates a MeanSquaredError
     *
     * @param name the name for the loss function
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanSquaredError(String name, Reduction reduction) {
        super(name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Ops tf, Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        tf = tf.withSubScope(this.getName());
        Operand losses = Losses.mean_squared_error(tf, labels, predictions);
        return super.computeWeightedLoss(tf, losses, getReduction(), sampleWeights);
    }
    
    

}
