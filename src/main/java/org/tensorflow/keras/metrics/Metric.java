/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.NoOp;
import org.tensorflow.op.core.Variable;

/**
 *
 * @author Jim Clarke
 */
public abstract class Metric implements  MetricInterface {
    public static final double EPSILON = 1e-7;
    public static final float EPSILON_F = 1e-7F;

    protected final Reduction reduction;
    protected final Ops tf;
    protected final String name;
    protected final DataType dType;
    
    // for graph mode
    protected final Graph graph;
    
    protected Map<String, MetricVariable> variables = new HashMap<>();
    
    protected  boolean stateful = true;
    protected boolean built = true;
    
    
    /**
     * create a metric with  name = class name and reduction = AUTO
     * 
     * @param tf the TensorFlow Ops when using Eager Mode
     */
    protected Metric(Ops tf) {
        this(tf, null, Reduction.SUM);
    }
    
    
    

    /**
     * create a metric with reduction = AUTO 
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param name the name of the metric
     */
    protected Metric(Ops tf, String name) {
        this(tf, name, Reduction.SUM);
    }
    
    
    
    
    /**
     * create a metric
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param reduction the reduction
     */
    protected Metric(Ops tf, Reduction reduction) {
        this(tf, null,reduction, null);
    }
    
   
    
    
    /**
     * create a metric
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param reduction the reduction
     * @param dType the DataType to use
     */
    protected Metric(Ops tf, Reduction reduction, DataType dType) {
        this(tf, null,reduction, dType);
    }
    
    
    
    
    /**
     * create a metric 
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param name the name of the metric
     * @param reduction the reduction
     */
    protected Metric(Ops tf, String name, Reduction reduction) {
        this(tf, name, reduction, null);
    }
    
    
    
    /**
     * create a metric 
     * @param tf the TensorFlow ops
     * @param name the name of this metric
     * @param reduction the reduction
     * @param dType the DataType
     */
    protected Metric(Ops tf, String name, Reduction reduction, DataType dType) {
        this.dType = dType;
        this.name = name == null ? this.getClass().getSimpleName() : name;
        this.reduction = reduction;
        this.tf = tf != null ? tf.withSubScope(this.name) : null;
        if(this.tf != null && this.tf.scope().env() instanceof Graph) {
            this.graph = (Graph)tf.scope().env();
        }else {
            this.graph = null;
        }
    }
    
   
    
    /**
     * add a variable to be collect metric values
     * 
     * @param name the name of the variable 
     * @param variable the variable 
     */
    protected void addVariable(String name, Variable variable) {
        this.variables.put(name, new MetricVariable(tf, name, variable));
    }
    
    /**
     * add a variable to be collect metric values
     * 
     * @param name the name of the variable 
     * @param variable the variable 
     * @param initializer the variable initializer
     */
    protected void addVariable(String name,  Variable variable, Initializer initializer) {
        this.variables.put(name, new MetricVariable(tf, name, variable, initializer));
    }
    
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Op resetStates() {
        List<Op> updateOperations = new ArrayList<>();
        this.variables.values().forEach((v) -> {
            updateOperations.add(tf.assign(v.getVariable(),v.initialize()));
        });
        Scope scope = tf.scope().withSubScope("resetStates");
        scope = scope.withControlDependencies(updateOperations);
        return NoOp.create(scope);
        
    }
    
    public  Variable getVariable(String name) {
       MetricVariable mv = variables.get(name);
       return mv != null ? mv.getVariable() : null;
    }

    /**
     * @return the reduction
     */
    public Reduction getReduction() {
        return reduction;
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

    
        
}
