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
import java.util.WeakHashMap;
import org.tensorflow.DataType;
import org.tensorflow.ExecutionEnvironment;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.keras.backend.tf.ControlDependencies;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TType;

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

    /**
     * variables are stored by Scope, and then by an identifier name
     */
    protected static Map<ExecutionEnvironment, Map<String, MetricVariable>> variables = new WeakHashMap<>();


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
        assert tf.scope().env().isGraph() : "Metric class has to be executed in Graph Mode";
        this.dType = dType == null ? TFloat32.DTYPE : dType;
        this.name = name == null ? this.getClass().getSimpleName() : name;
        this.tf = tf.withSubScope(this.name);
    }
    
    /**
     * {@inheritDoc}
    */
    public Op updateState(Operand... args) {
        List<Op> conrolOps = updateStateList(args);
        return ControlDependencies.addControlDependencies(tf, name + "_updateState", conrolOps);
    }
    
    /**
     * {@inheritDoc}
    */
    public Operand result() {
        return this.result(this.tf);
    }

    /**
     * Call update state followed by a call to result
     *
     * @param args the args to be passed to update state
     * @return the result with a control dependency on update state
     */
    public Operand callOnce(Operand... args) {
        List<Op> conrolOps = new ArrayList<>();
        conrolOps.addAll(updateStateList(args));
        return ControlDependencies.addControlDependencies(tf, (tf) -> result(tf), name + "_call", conrolOps);
    }
    
    protected String getVariableName(String id) {
        return String.format("%s_%s_%s", this.getClass().getSimpleName(),
                this.name, id);
    }

    /**
     * add a variable to be collect metric values
     *
     * @param name a name that identifies the variable
     * @param variable the variable
     */
    protected void addVariable(String name, Variable variable) {
        Map<String, MetricVariable> thisMap = this.variables.get(tf.scope().env());
        if(thisMap == null) {
            thisMap = new HashMap<>();
            variables.put(tf.scope().env(), thisMap);
        }
        thisMap.put(name, new MetricVariable(tf, name, variable));
    }

    /**
     * add a variable to be collect metric values
     *
     * @param name a name that identifies the variable.
     * @param variable the variable
     * @param initializer the variable initializer
     */
    protected void addVariable(String name, Variable variable, Initializer initializer) {
        Map<String, MetricVariable> thisMap = variables.get(tf.scope().env());
        if(thisMap == null) {
            thisMap = new HashMap<>();
            variables.put(tf.scope().env(), thisMap);
        }
        thisMap.put(name, new MetricVariable(tf, name, variable, initializer));
    }

    public List<Variable> getVariables() {
        Map<String, MetricVariable> thisMap = this.variables.get(tf.scope().env());
        List<Variable> result = new ArrayList<>();
        if(thisMap != null) {
            thisMap.values().forEach(mv -> result.add(mv.getVariable()));
        }
        return result;
    }
    
    public Op initializeVars() {
        return initializeVars("initializeVars");
    }
    
    
    private List<Op> initializeVarsList(String subScopeName) {
        Map<String, MetricVariable> thisMap = this.variables.get(tf.scope().env());
        List<Op> updateOperations = new ArrayList<>();
        if(thisMap != null) {
            thisMap.values().forEach((v) -> 
                updateOperations.add(tf.assign(v.getVariable(), v.initialize()))
            );
        }
        return updateOperations;
    }
    
    public Op initializeVars(String subScopeName) {
        Map<String, MetricVariable> thisMap = this.variables.get(tf.scope().env());
        
        List<Op> updateOperations = initializeVarsList(subScopeName);
        return ControlDependencies.addControlDependencies(tf, subScopeName, updateOperations);
    }
    
    /**
     * Adds a value to a Variable, making sure it has been initialized first.
     * 
     * @param name a name that identifies the variable.
     * @param variable the variable
     * @param val the value to assign to the variable
     * @return the variable add operation with necessary control dependencies
     * @param <T> the type of Operand
     */
    public <T extends TType> Operand<T> variableAssignAdd(String name, Variable variable,  Operand<T> val) {
         Map<String, MetricVariable> thisMap = Metric.variables.get(tf.scope().env());
         if(thisMap == null) {
             tf.assignAdd(variable, val);
         }
         
         MetricVariable v = thisMap.get(name);
         if(v != null) {
             if(v.isInitialized()) {
                 return tf.assignAdd(variable, val);
             }else {
                Operand<T> assign = tf.assign(variable, v.initialize());
                Operand<T> assignAdd =  ControlDependencies.addControlDependencies(
                        tf, (tf1)->tf1.assignAdd(variable, val),
                        "var_init", assign);
                v.setInitialized(true);
                return assignAdd;
             }
         }else {
             return tf.assignAdd(variable, val);
         }
         
    }
        

    /**
     * {@inheritDoc}
     */
    @Override
    public Op resetStates() {
         return initializeVars("resetStates");

    }

    public Variable getVariable(String name) {
        Map<String, MetricVariable> thisMap = this.variables.get(tf.scope().env());
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
