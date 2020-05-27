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
    public SquaredHinge(Ops tf) {
        super(tf, "squared_hinge");
    }
    
    /**
     * Creates a Poisson
     * @param reduction Type of Reduction to apply to loss.
     */
    public SquaredHinge(Ops tf, Reduction reduction) {
        super(tf, "squared_hinge", reduction);
    }
    
    /**
     * Creates a Poisson
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public SquaredHinge(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.squared_hinge(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
