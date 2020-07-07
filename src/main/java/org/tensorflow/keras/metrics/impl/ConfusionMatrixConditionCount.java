/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.metrics.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.tf.ControlDependencies;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public abstract class ConfusionMatrixConditionCount extends Metric {
    public static final String ACCUMULATOR = "accumulator";
    
    private Variable<TFloat32> accumulator;
    private Assign initializer;
    private boolean initialized = false;
    
    private ConfusionMatrixEnum confusion_matrix_cond;
    private final float[] thresholds;
    
    private Map<ConfusionMatrixEnum, Variable> confusionMatrix = new HashMap<>();
    
    
    public ConfusionMatrixConditionCount(Ops tf, String name, ConfusionMatrixEnum confusion_matrix_cond) {
        this(tf, name,confusion_matrix_cond,  new float[]{0.5f}, null);
    }
    public ConfusionMatrixConditionCount(Ops tf, String name, ConfusionMatrixEnum confusion_matrix_cond, float threshold) {
        this(tf, name,confusion_matrix_cond,  new float[] { threshold}, null );
    }
    public ConfusionMatrixConditionCount(Ops tf, String name, ConfusionMatrixEnum confusion_matrix_cond, float[] thresholds, DataType dType) {
        super(tf, name, dType);
        this.confusion_matrix_cond = confusion_matrix_cond;
        this.thresholds = thresholds;
        init();
    }
    
    private void init() {
        Shape variableShape =  Shape.of(this.thresholds.length);
        
        accumulator = getVariable(ACCUMULATOR);
        if(accumulator == null) {
            Zeros zeros = new Zeros(tf);
            accumulator = tf.withName(ACCUMULATOR).variable(
                    zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            initializer = tf.assign(accumulator, zeros.call(tf.constant(variableShape), TFloat32.DTYPE));
            this.addVariable(ACCUMULATOR, accumulator, zeros);
        }
        
        confusionMatrix.put(confusion_matrix_cond, accumulator);
    }
    

    @Override
    public Op updateState(Operand... operands) {
        Operand yTrue = operands[0];
        Operand yPred = operands[1];
        Operand sampleWeights = operands.length > 2 ? operands[2] : null;
        List<Op> updateOperations = new ArrayList<>();
        updateOperations.addAll(Metrics.update_confusion_matrix_variables(tf,
                confusionMatrix,
                Collections.EMPTY_MAP,
                yTrue,
                yPred, 
                this.thresholds,
                null,
                null,
                sampleWeights, 
                false, 
                null));
        
        return ControlDependencies.addControlDependencies(tf,
                "updateState", updateOperations);
    }

    @Override
    public Operand result() {
        return tf.identity(accumulator);
    }
    
    /**
     * get the thresholds
     * @return the thresholds
     */
    public float[] getThresholds() {
        return this.thresholds;
    }
    
}
