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
 * Computes the cross-entropy loss between true labels and predicted labels.
 *
 * @author Jim Clarke
 */
public class BinaryCrossentropy extends Loss {

    private final boolean fromLogits;
    private final float labelSmoothing;

    /**
     * Creates a BinaryCrossentropy with Reduction.AUTO, labelSmoothing=0.0 and
     * fromLogits=false.
     */
    public BinaryCrossentropy() {
        this("binary_crossentropy", false, 0.0F, Reduction.AUTO);
    }

    /**
     * Creates a BinaryCrossentropy with Reduction.AUTO, labelSmoothing=0.0 and
     * fromLogits=false.
     *
     * @param name the name of this loss function
     */
    public BinaryCrossentropy(String name) {
        this(name, false, 0.0F, Reduction.AUTO);
    }

    /**
     * Creates a BinaryCrossentropy with labelSmoothing = 0.0 and
     * fromLogits=false.
     *
     * @param reduction Type of Reduction to apply to loss.
     */
    public BinaryCrossentropy(Reduction reduction) {
        this("binary_crossentropy", false, 0.0F, reduction);
    }

    /**
     * Creates a BinaryCrossentropy with labelSmoothing = 0.0 and
     * fromLogits=false.
     *
     * @param name the name of this loss function
     * @param reduction Type of Reduction to apply to loss.
     */
    public BinaryCrossentropy(String name, Reduction reduction) {
        this(name, false, 0.0F, reduction);
    }

     /**
     * Creates a BinaryCrossentropy with Reduction.AUTO,
     *
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public BinaryCrossentropy(boolean fromLogits) {
        this("binary_crossentropy", fromLogits, 0.0F, Reduction.AUTO);
    }
    
     /**
     * Creates a BinaryCrossentropy with Reduction.AUTO,
     *
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public BinaryCrossentropy(String name, boolean fromLogits) {
        this(name, fromLogits, 0.0F, Reduction.AUTO);
    }
    
    /**
     * Creates a BinaryCrossentropy with Reduction.AUTO,
     *
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public BinaryCrossentropy(boolean fromLogits, float labelSmoothing) {
        this("binary_crossentropy", fromLogits, labelSmoothing, Reduction.AUTO);
    }

    /**
     * Creates a BinaryCrossentropy with Reduction.AUTO,
     *
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public BinaryCrossentropy(String name, boolean fromLogits, float labelSmoothing) {
        this(name, fromLogits, labelSmoothing, Reduction.AUTO);
    }

    /**
     * Creates a BinaryCrossentropy
     *
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @param reduction Type of Reduction to apply to loss.
     */
    public BinaryCrossentropy(boolean fromLogits, float labelSmoothing, Reduction reduction) {
        this("binary_crossentropy", fromLogits, labelSmoothing, reduction);
    }

    /**
     * Creates a BinaryCrossentropy
     *
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @param reduction Type of Reduction to apply to loss.
     */
    public BinaryCrossentropy(String name, boolean fromLogits, float labelSmoothing, Reduction reduction) {
        super(name, reduction);
        this.fromLogits = fromLogits;
        this.labelSmoothing = labelSmoothing;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Ops tf, Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        tf = tf.withSubScope(this.getName());
        Operand losses = Losses.binary_crossentropy(tf, labels, predictions, fromLogits, labelSmoothing);
        return super.computeWeightedLoss(tf, losses, getReduction(), sampleWeights);
    }

}
