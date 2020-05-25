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
 * Computes the categorical hinge loss between labels and [redictions.
 * @author Jim Clarke
 */
public class CategoricalHinge extends Loss {
    /**
     * Creates a CategoricalHinge with Reduction.AUTO
     */
    public CategoricalHinge() {
        super("categorical_hinge");
    }
    
    /**
     * Creates a CategoricalHinge with Reduction.AUTO
     * 
     * @param name the name of the loss
     */
    public CategoricalHinge(String name) {
        super(name);
    }
    
    /**
     * Creates a CategoricalHinge
     * @param reduction Type of Reduction to apply to loss.
     */
    public CategoricalHinge(Reduction reduction) {
        super("categorical_hinge", reduction);
    }
    
    /**
     * Creates a CategoricalHinge
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public CategoricalHinge(String name, Reduction reduction) {
        super(name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Ops tf,  Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        tf = tf.withSubScope(this.getName());
        Operand losses = Losses.categorical_hinge(tf, labels, predictions);
        return super.computeWeightedLoss(tf, losses, getReduction(), sampleWeights);
    }
    
}
