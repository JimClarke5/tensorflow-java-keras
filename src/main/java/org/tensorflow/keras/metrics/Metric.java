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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.keras.backend.tf.ControlDependencies;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public abstract class Metric implements MetricInterface {

    public static final double EPSILON = 1e-7;
    public static final float EPSILON_F = 1e-7F;

    protected final Ops tf;
    protected final String name;
    protected final DataType dType;

    // for graph mode
    protected static Graph graph;
    
    protected static Map<String, Map<String, MetricVariable>> variables = new HashMap<>();


    protected boolean stateful = true;
    protected boolean built = true;

    // for debug
    private Session session;

    /**
     * create a metric with name = class name and reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     */
    protected Metric(Ops tf) {
        this(tf, null, null);
    }

    /**
     * create a metric with reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param name the name of the metric
     */
    protected Metric(Ops tf, String name) {
        this(tf, name, null);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param dType the DataType to use
     */
    protected Metric(Ops tf, DataType dType) {
        this(tf, null, dType);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow ops
     * @param name the name of this metric
     * @param dType the DataType
     */
    protected Metric(Ops tf, String name, DataType dType) {
        assert tf.scope().env().isGraph() : "Metric class have to be executed in Graph Mode";
        this.dType = dType == null ? TFloat32.DTYPE : dType;
        this.name = name == null ? this.getClass().getSimpleName() : name;
        this.tf = tf.withSubScope(this.name);
        if (this.tf != null && this.tf.scope().env() instanceof Graph) {
            if (graph != null && !graph.equals((Graph) this.tf.scope().env())) {
                graph = (Graph) tf.scope().env();
                variables.clear();
            } else if (graph == null) {
                graph = (Graph) tf.scope().env();
            }
        } else {
            graph = null;
        }
    }

    /**
     * Call update state followed by a call to result
     *
     * @param args the args to be passed to update state
     * @return the result with a control dependency on update state
     */
    public Operand call(Operand... args) {
        Op op = updateState(args);
        return ControlDependencies.addControlDependencies(tf, (tf) -> result(), name + "/call", op);
    }

    /**
     * add a variable to be collect metric values
     *
     * @param name the name of the variable
     * @param variable the variable
     */
    protected void addVariable(String name, Variable variable) {
        Map<String, MetricVariable> thisMap = this.variables.get(this.name);
        if(thisMap == null) {
            thisMap = new HashMap<>();
        }
        thisMap.put(name, new MetricVariable(tf, name, variable));
    }

    /**
     * add a variable to be collect metric values
     *
     * @param name the name of the variable
     * @param variable the variable
     * @param initializer the variable initializer
     */
    protected void addVariable(String name, Variable variable, Initializer initializer) {
        Map<String, MetricVariable> thisMap = variables.get(this.name);
        if(thisMap == null) {
            thisMap = new HashMap<>();
            variables.put(this.name, thisMap);
        }
        thisMap.put(name, new MetricVariable(tf, name, variable, initializer));
    }

    public List<Variable> getVariables() {
        Map<String, MetricVariable> thisMap = this.variables.get(this.name);
        List<Variable> result = new ArrayList<>();
        if(thisMap != null) {
            thisMap.values().forEach(mv -> result.add(mv.getVariable()));
        }
        return result;
    }
    public Op initializeVars() {
        return initializeVars("initializeVars");
    }
    
    public Op initializeVars(String subScopeName) {
        Map<String, MetricVariable> thisMap = this.variables.get(this.name);
        
        List<Op> updateOperations = new ArrayList<>();
        if(thisMap != null) {
            thisMap.values().forEach((v) -> 
                updateOperations.add(tf.assign(v.getVariable(), v.initialize()))
            );
        }
        return ControlDependencies.addControlDependencies(tf, subScopeName, updateOperations);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Op resetStates() {
         return initializeVars("resetStates");

    }

    public Variable getVariable(String name) {
        Map<String, MetricVariable> thisMap = this.variables.get(this.name);
        if(thisMap == null) return null;
        MetricVariable mv = thisMap.get(name);
        return mv != null ? mv.getVariable() : null;
    }

    /**
     * @return the tf
     */
    public Ops getTF() {
        return tf;
    }

    /**
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     *
     * @return the dtype
     */
    public DataType getDataType() {
        return dType;
    }

    public boolean isDebug() {
        return this.session != null;
    }

    /**
     * @return the session
     */
    public Session getSession() {
        return session;
    }

    /**
     * @param session the session to set
     */
    public void setDebug(Session session) {
        this.session = session;
        Metrics.setDebug(session);
    }
    
    public void resetDebug() {
        setDebug(null);
    }

}
