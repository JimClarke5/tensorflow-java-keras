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
 * Computes Computes the logarithm of the hyperbolic cosine of the prediction error..
 * @author Jim Clarke
 */
public class LogCosh extends Loss {
    /**
     * Creates a LogCosh with Reduction.AUTO
     */
    public LogCosh(Ops tf) {
        super(tf, "logcosh");
    }
    
    /**
     * Creates a LogCosh
     * @param reduction Type of Reduction to apply to loss.
     */
    public LogCosh(Ops tf, Reduction reduction) {
        super(tf, "logcosh", reduction);
    }
    
    /**
     * Creates a LogCosh
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public LogCosh(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.logcosh(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }
    
}
