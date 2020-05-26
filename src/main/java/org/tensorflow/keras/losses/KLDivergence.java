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
 * Computes Computes Kullback-Leibler divergence loss between labels and predictions.
 * @author Jim Clarke
 */
public class KLDivergence extends Loss {
    /**
     * Creates a KLDivergence with Reduction.AUTO
     */
    public KLDivergence() {
        super("kullback_leibler_divergence");
    }
    
    /**
     * Creates a KLDivergence
     * @param reduction Type of Reduction to apply to loss.
     */
    public KLDivergence(Reduction reduction) {
        super("kullback_leibler_divergence", reduction);
    }
    
    /**
     * Creates a KLDivergence
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public KLDivergence(String name, Reduction reduction) {
        super(name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Ops tf,  Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        tf = tf.withSubScope(this.getName());
        Operand losses = Losses.kullback_leibler_divergence(tf, labels, predictions);
        return super.computeWeightedLoss(tf, losses, getReduction(), sampleWeights);
    }
    
}
