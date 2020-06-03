/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics;

import org.tensorflow.DataType;
import org.tensorflow.keras.losses.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.metrics.impl.MeanMetricWrapper;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Computes the cross-entropy loss between true labels and predicted labels.
 *
 * @author Jim Clarke
 */
public class SparseCategoricalCrossentropy extends MeanMetricWrapper implements LossFunction {
    
    public static final String DEFAULT_NAME = "sparse_categorical_crossentropy";

    private final boolean fromLogits;
    private final float labelSmoothing;
    private final int axis;

    /**
     * Creates a SparseCategoricalCrossentropy labelSmoothing=0.0 and
     * fromLogits=false.
     * 
     * @param tf the TensorFlow Ops
     */
    public SparseCategoricalCrossentropy(Ops tf) {
        this(tf, DEFAULT_NAME, false, 0.0F, -1, null);
    }
    
    public SparseCategoricalCrossentropy(Ops tf, int axis) {
        this(tf, DEFAULT_NAME, false, 0.0F, axis, null);
    }

    /**
     * Creates a SparseCategoricalCrossentropy labelSmoothing=0.0 and
     * fromLogits=false.
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     */
    public SparseCategoricalCrossentropy(Ops tf, String name) {
        this(tf, name, false, 0.0F, -1, null);
    }
    
    public SparseCategoricalCrossentropy(Ops tf, String name, int axis) {
        this(tf, name, false, 0.0F, axis, null);
    }


     /**
     * Creates a SparseCategoricalCrossentropy 
     *
     * @param tf the TensorFlow Ops
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public SparseCategoricalCrossentropy(Ops tf, boolean fromLogits) {
        this(tf, DEFAULT_NAME, fromLogits, 0.0F, -1, null);
    }
    public SparseCategoricalCrossentropy(Ops tf, boolean fromLogits, int axis) {
        this(tf, DEFAULT_NAME, fromLogits, 0.0F, axis, null);
    }
    
     /**
     * Creates a SparseCategoricalCrossentropy 
     *
     * @param tf the TensorFlow Ops
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public SparseCategoricalCrossentropy(Ops tf, String name, boolean fromLogits) {
        this(tf, name, fromLogits, 0.0F, -1, null);
    }
    
    public SparseCategoricalCrossentropy(Ops tf, String name, boolean fromLogits, int axis) {
        this(tf, name, fromLogits, 0.0F, axis, null);
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
     */
    public SparseCategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing) {
        this(tf, DEFAULT_NAME, fromLogits, labelSmoothing, -1, null);
    }
    
    public SparseCategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing, int axis) {
        this(tf, DEFAULT_NAME, fromLogits, labelSmoothing, axis, null);
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
     */
    public SparseCategoricalCrossentropy(Ops tf, String name, boolean fromLogits, float labelSmoothing, int axis, DataType dType) {
        super(tf, name, dType);
        this.fromLogits = fromLogits;
        this.labelSmoothing = labelSmoothing;
        this.axis = axis;
        super.setLoss(this);
    }

    /**
     * @return the fromLogits
     */
    public boolean isFromLogits() {
        return fromLogits;
    }

    /**
     * @return the labelSmoothing
     */
    public float getLabelSmoothing() {
        return labelSmoothing;
    }
    
    /**
     * @return the labelSmoothing
     */
    public int getAxis() {
        return axis;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.sparse_categorical_crossentropy(tf, labels, predictions, isFromLogits(), axis);
        return losses;
    }

}
