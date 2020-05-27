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
 * Computes the Poisson loss between the labels and predictions
 * @author Jim Clarke
 */
public class SquaredHinge extends Loss {
    /**
     * Creates a Poisson with Reduction.AUTO
     */
    public SquaredHinge() {
        super("squared_hinge");
    }
    
    /**
     * Creates a Poisson
     * @param reduction Type of Reduction to apply to loss.
     */
    public SquaredHinge(Reduction reduction) {
        super("squared_hinge", reduction);
    }
    
    /**
     * Creates a Poisson
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public SquaredHinge(String name, Reduction reduction) {
        super(name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Ops tf,  Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        tf = tf.withSubScope(this.getName());
        Operand losses = Losses.squared_hinge(tf, labels, predictions);
        return super.computeWeightedLoss(tf, losses, getReduction(), sampleWeights);
    }
    
}
