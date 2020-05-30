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
public class SparseCategoricalCrossentropy extends Loss {
    
    public static final String DEFAULT_NAME = "sparse_categorical_crossentropy";

    private final boolean fromLogits;
    private final float labelSmoothing;

    /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO, labelSmoothing=0.0 and
     * fromLogits=false.
     * 
     * @param tf the TensorFlow Ops
     */
    public SparseCategoricalCrossentropy(Ops tf) {
        this(tf, DEFAULT_NAME, false, 0.0F, Reduction.AUTO);
    }

    /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO, labelSmoothing=0.0 and
     * fromLogits=false.
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     */
    public SparseCategoricalCrossentropy(Ops tf, String name) {
        this(tf, name, false, 0.0F, Reduction.AUTO);
    }

    /**
     * Creates a SparseCategoricalCrossentropy with labelSmoothing = 0.0 and
     * fromLogits=false.
     *
     * @param tf the TensorFlow Ops
     * @param reduction Type of Reduction to apply to loss.
     */
    public SparseCategoricalCrossentropy(Ops tf, Reduction reduction) {
        this(tf, DEFAULT_NAME, false, 0.0F, reduction);
    }

    /**
     * Creates a SparseCategoricalCrossentropy with labelSmoothing = 0.0 and
     * fromLogits=false.
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     * @param reduction Type of Reduction to apply to loss.
     */
    public SparseCategoricalCrossentropy(Ops tf, String name, Reduction reduction) {
        this(tf, name, false, 0.0F, reduction);
    }

     /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO,
     *
     * @param tf the TensorFlow Ops
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public SparseCategoricalCrossentropy(Ops tf, boolean fromLogits) {
        this(tf, DEFAULT_NAME, fromLogits, 0.0F, Reduction.AUTO);
    }
    
     /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO,
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public SparseCategoricalCrossentropy(Ops tf, String name, boolean fromLogits) {
        this(tf, name, fromLogits, 0.0F, Reduction.AUTO);
    }
    
    /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO,
     *
     * @param tf the TensorFlow Ops
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public SparseCategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing) {
        this(tf, DEFAULT_NAME, fromLogits, labelSmoothing, Reduction.AUTO);
    }

    /**
     * Creates a SparseCategoricalCrossentropy with Reduction.AUTO,
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public SparseCategoricalCrossentropy(Ops tf, String name, boolean fromLogits, float labelSmoothing) {
        this(tf, name, fromLogits, labelSmoothing, Reduction.AUTO);
    }

    /**
     * Creates a SparseCategoricalCrossentropy
     *
     * @param tf the TensorFlow Ops
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @param reduction Type of Reduction to apply to loss.
     */
    public SparseCategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing, Reduction reduction) {
        this(tf, DEFAULT_NAME, fromLogits, labelSmoothing, reduction);
    }

    /**
     * Creates a SparseCategoricalCrossentropy
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     * @param reduction Type of Reduction to apply to loss.
     */
    public SparseCategoricalCrossentropy(Ops tf, String name, boolean fromLogits, float labelSmoothing, Reduction reduction) {
        super(tf, name, reduction);
        this.fromLogits = fromLogits;
        this.labelSmoothing = labelSmoothing;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.sparse_categorical_crossentropy(tf, labels, predictions, fromLogits);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }

}
