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
    public KLDivergence(Ops tf) {
        super(tf, "kullback_leibler_divergence");
    }
    
    /**
     * Creates a KLDivergence
     * @param reduction Type of Reduction to apply to loss.
     */
    public KLDivergence(Ops tf, Reduction reduction) {
        super(tf, "kullback_leibler_divergence", reduction);
    }
    
    /**
     * Creates a KLDivergence
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public KLDivergence(Ops tf, String name, Reduction reduction) {
        super(tf,name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.kullback_leibler_divergence(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
