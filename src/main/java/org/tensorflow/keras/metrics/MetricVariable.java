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
package org.tensorflow.keras.metrics;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.initializers.Constant;
import org.tensorflow.keras.initializers.GlorotUniform;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.utils.TypeUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Variable;
import org.tensorflow.proto.framework.VariableAggregation;
import org.tensorflow.proto.framework.VariableSynchronization;

/**
 *
 * @author Jim Clarke
 */
public class MetricVariable {

    private final String name;
    private final Variable variable;
    private final VariableAggregation aggregation;
    private final VariableSynchronization synchronization;
    private final Initializer initializer;
    private final Ops tf;

    public MetricVariable(Ops tf, String name, Variable variable) {
        this(tf, name, variable, VariableAggregation.VARIABLE_AGGREGATION_SUM,
                VariableSynchronization.VARIABLE_SYNCHRONIZATION_ON_READ, null);
    }

    public MetricVariable(Ops tf, String name, Variable variable, Initializer initializer) {
        this(tf, name, variable, VariableAggregation.VARIABLE_AGGREGATION_SUM,
                VariableSynchronization.VARIABLE_SYNCHRONIZATION_ON_READ, initializer);
    }

    public MetricVariable(Ops tf, String name, Variable variable, VariableAggregation aggregation,
            VariableSynchronization synchronization, Initializer initializer) {
        this.tf = tf;
        this.name = name;
        this.variable = variable;
        this.aggregation = aggregation;
        this.synchronization = synchronization;
        DataType dType = variable.asOutput().dataType();
        if (initializer == null) {
            if (TypeUtils.isFloating(dType)) {
                this.initializer = new GlorotUniform(tf);
            } else if (TypeUtils.isInteger(dType) || TypeUtils.isBoolean(dType)) {
                this.initializer = new Zeros(tf);
            } else {
                throw new IllegalArgumentException(
                        String.format("An initializer for variable %s of type %s is required",
                                variable.toString(), dType)
                );
            }
        } else {
            this.initializer = initializer;
        }
    }

    public MetricVariable(Ops tf, String name, Variable variable, VariableAggregation aggregation,
            VariableSynchronization synchronization, Object initializer) {
        this(tf, name, variable, aggregation, synchronization, Initializers.get(tf, initializer));
    }

    public Operand initialize() {

        Operand operand = initializer.call(
                tf.constant(variable.asOutput().shape()),
                variable.asOutput().dataType());

        return tf.assign(variable, operand, Assign.useLocking(Boolean.TRUE));
    }

    public Operand initialize(double scalar) {
        Constant initializer = new Constant(tf, scalar);
        Operand operand = initializer.call(
                tf.constant(variable.asOutput().shape()),
                variable.asOutput().dataType());

        return tf.assign(variable, operand, Assign.useLocking(Boolean.TRUE));
    }

    /**
     * @return the variable
     */
    public Variable getVariable() {
        return variable;
    }

    /**
     * @return the aggregation
     */
    public VariableAggregation getAggregation() {
        return aggregation;
    }

    /**
     * @return the synchronization
     */
    public VariableSynchronization getSynchronization() {
        return synchronization;
    }

    /**
     * @return the initializer
     */
    public Initializer getInitializer() {
        return initializer;
    }
}
