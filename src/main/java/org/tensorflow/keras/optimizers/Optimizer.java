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
package org.tensorflow.keras.optimizers;

import com.github.javaparser.utils.Pair;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.Tensor;
import org.tensorflow.keras.initializers.Initializer;
import org.tensorflow.keras.initializers.Initializers;
import org.tensorflow.keras.utils.Options;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.ClipByValue;
import org.tensorflow.op.core.Variable;
import org.tensorflow.proto.framework.VariableAggregation;
import org.tensorflow.proto.framework.VariableSynchronization;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.NdArray;
import org.tensorflow.types.TBfloat16;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat16;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;

/**
 *
 * @author Jim Clarke
 */
public abstract class Optimizer <T extends Number> {
    
    protected final String[] ALLOWED_KWARGS = {"clipnorm", "clipvalue", "lr", "decay"};
    
    private final String name;
    private final Options kwargs;
    
    boolean _use_locking;
    Map _hyper = new HashMap<>();
    Map<String, Map> _slots = new HashMap<>();
    List<String> _slot_names = new ArrayList<>();
    List<Number> _weights = new ArrayList<>();
    int _iterations = 0;
    Map<String, Map>  _deferred_slot_restorations = new HashMap<>();
    double decay = 0.0;
    double _initial_decay = 0.0;
    Double clipnorm = null;
    Double clipvalue = null;
    boolean _hypers_created = false;
    List<NdArray> weights;
    List<Variable> _variables;
    Map<Variable, Map<String, Slot>> slots = new HashMap<>();
    Ops tf = Ops.create();
    
    public Optimizer() {
        this(null, (Options)null);
    }
    
    public Optimizer(String name) {
        this(name, (Options)null);
    }
     
    public Optimizer(Options kwargs) {
        this(null, kwargs);
    }
    
    public Optimizer(String name, Options kwargs) {
        this.name = name;
        this.kwargs = kwargs;
        
        if(kwargs != null) {
            kwargs.validate(ALLOWED_KWARGS);
            if(kwargs.containsOption("decay")) {
                this.decay = (Double)kwargs.pop("decay");
                this._initial_decay = decay;
            }
            // Set the gradient clipping properties
            if(kwargs.containsOption("clipnorm")) {
                this.clipnorm = (Double)kwargs.pop("clipnorm");
            }
            this.clipvalue = (Double)kwargs.pop("clipvalue", null);
        }
        
    }
    /**
     * 
     * @param grad a `Tensor` representing the gradient.
     * @param handle a `Tensor` of dtype `resource` which points to the variable to be updated.
     * @param apply_state A dict which is used across multiple apply calls.
     * @return 
     */
    protected abstract Operation resource_apply_dense(Tensor grad, Tensor handle, Map apply_state);
    
    /**
     * 
     * @param grad a `Tensor` representing the gradient.
     * @param handle a `Tensor` of dtype `resource` which points to the variable to be updated.
     * @param indices a `Tensor` of integral type representing the indices for which
        the gradient is nonzero. Indices are unique.
     * @param apply_state A dict which is used across multiple apply calls.
     * @return 
     */
    protected abstract Operation resource_apply_sparse(Tensor grad, Tensor handle, Tensor indices, Map apply_state);
    
    /**
     *  
     * @return Returns the config of the optimizer.
     */
    protected  Map<String, Object> getConfig() {
        return getConfig(new HashMap<String, Object>());
    }
    /**
     * @param config the config 
     * @return Returns the config of the optimizer.
     */
    protected  Map<String, Object> getConfig(Map<String, Object> config) {
        config.put("name", this.name);
        if(this.clipnorm != null) {
            config.put("clipnorm", this.clipnorm);
        }
        if(this.clipvalue != null) {
            config.put("clipvalue", this.clipvalue);
        }
        return config;
    }
    
    
    protected void add_slot(Variable variable, String name) {
        add_slot(variable, name, "zeros");
    }
    
    protected void add_slot(Variable variable, String name, Object initializer) {
        Slot slot = new Slot(name, variable, Initializers.get(initializer));
        
        Map<String, Slot> slotMap = slots.get(variable);
        if(slotMap == null) {
            slotMap = new HashMap<>();
            slots.put(variable, slotMap);
        }
        slotMap.put(name, slot);
        if(!_slot_names.contains(name)) {
            _slot_names.add(name);
        }
    }
    
    protected Slot get_slot(Variable variable, String name) {
        Map<String, Slot> slotMap = slots.get(variable);
        return slotMap != null? slotMap.get(name) : null;
    }
    
    protected List<String> getSlotNames() {
        return _slot_names;
    }
    
    public Operation getUpdates(Tensor loss, List<Variable> params) {
        List<Tensor> grads = this.getGradients(loss, params);
        List<Pair> grads_and_vars = zip(grads, params);
        //this._assert_valid_dtypes([
        //        v for g, v in grads_and_vars
        //        if g is not None and v.dtype != dtypes.resource
        //    ])
        return this.apply_gradients(grads_and_vars);
    }
    
    private List<Pair> zip(List a, List b) {
        assert(a.size() == b.size());
        List<Pair> result = new ArrayList<>();
        for(int i = 0; i < a.size(); i++) {
            result.add(new Pair(a.get(i), b.get(i)));
        }
        return result;
    }
    

    
    public void add_weight(
            String name, Shape shape, DataType dtype, 
            Object initializer, boolean trainable,
            VariableSynchronization synchronization,
            VariableAggregation aggregation
        ){
        Initializer linitializer = initializer != null ? 
                Initializers.get(initializer) : Initializers.get("zeros"); 
        
        Operand fl = linitializer.call(tf, tf.constant(shape), dtype);
        
    }
    
    /**
     * 
     * @return 
     */
    public List<NdArray> get_weights() {
        return this.weights;
    }
    
    /**
     * 
     * @param weights 
     */
    public void setWeights(List<NdArray> weights) {
        this.weights = weights;
    }
    
    /**
     * 
     * @return 
     */
    public List<Variable> variables() {
        return this._variables;
    }
    
    // loss, var_list, grad_loss=None, name=None
    /**
     * 
     * @param loss A callable taking no arguments which returns the value to minimize.
     * @param var_list list of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` since the variables are created at the first time `loss` is
        called.
     * @param grad_loss Optional. A `Tensor` holding the gradient computed for `loss`.
     * @param name Optional name for the returned operation.
     * @return An `Operation` that updates the variables in `var_list`. The `iterations`
      will be automatically increased by 1.
     */
    public  Operation minimize(Supplier loss, Object var_list, 
            Tensor grad_loss, String name) {
        List<Pair> grads_and_vars = this._compute_gradients(
                loss, var_list, grad_loss);
        return this.apply_gradients(grads_and_vars, name);
    }
    
    public Operation apply_gradients(List<Pair> grads_and_vars) {
        return apply_gradients(grads_and_vars, null, true);
    }
    public Operation apply_gradients(List<Pair> grads_and_vars, String name) {
        return apply_gradients(grads_and_vars, name, true);
    }
    public Operation apply_gradients(
            List<Pair> grads_and_vars, String name, boolean all_reduce_sum_gradients) {
        if(name == null)
            name = this.name;
        return null;
    }
    
    
    
    public abstract List<Tensor> getGradients(Tensor loss, List<Variable> params);

    /**
     * Compute gradients of `loss` for the variables in `var_list`
     * 
     * @param loss A callable taking no arguments which returns the value to minimize.
     * @param var_list list of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` and the variables are created at the first time when `loss`
        is called.
     * @param grad_loss A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.
     * @return Optional. A `Tensor` holding the gradient computed for `loss`.
     */
    private List<Pair> _compute_gradients(Supplier<Double> loss, Object var_list, Tensor grad_loss) {
         List<Variable> varList;
        if(var_list != null) {
            if(var_list instanceof List) {
                List list = (List)var_list;
                varList = list;
            } else if(var_list instanceof Supplier) {
                Supplier<List<Variable>> slist = (Supplier<List<Variable>>)var_list;
                Supplier<List<Variable>> svlist = (Supplier<List<Variable>>)slist;
                varList = svlist.get();
            }
            else {
                throw new IllegalArgumentException("var_list muse either be a List<Variable> or Supplier<List<Variable>>");
            }
        } else {
            throw new NullPointerException("var_list must be set");
        }
        double loss_value = loss.get();
        Ops tf = Ops.create();
        Scope scope = tf.scope().withName(this.name + "/gradients");
        //List<Double> grads = tape.gradient(loss_value, var_list, grad_loss);
        //grads = this._clip_gradients(grads);   
        return null;
    }
    
    private Operand _clip_gradients(Operand<TFloat64> grads) {
        //"""Clip gradients according to the clipnorm and clipvalue attributes."""
        if (this.clipnorm != null) {
            /********************* TODO
            if distribute_ctx.has_strategy():
              raise ValueError("Gradient clipping in the optimizer "
                               "(by setting clipnorm or clipvalue) is currently "
                               "unsupported when using a distribution strategy.")
            ****************/
          //ClipByNorm<TFloat64> clipNorm = 
          //        ClipByNorm.create(tf.scope(), grads, tf.constant(clipnorm));
          //grads = clipNorm.asOutput();
          //grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
        if(this.clipvalue != null) {
          /************ TODO 
          if distribute_ctx.has_strategy():
            raise ValueError("Gradient clipping in the optimizer "
                             "(by setting clipnorm or clipvalue) is currently "
                             "unsupported when using a distribution strategy.")
          **************************/
          //  ClipByNorm<T extends TType>.create(Scope scope, Operand<T> t, Operand<T> clipValueMin, Operand<T> clipValueMax
          ClipByValue<TFloat64> clipValue = 
                  ClipByValue.create(tf.scope(), grads, tf.constant(-this.clipvalue), tf.constant(this.clipvalue));
          grads = clipValue.asOutput();
          
        }
        }
        return grads;
    }
    
        
        
    
    protected class Slot {

        /**
         * @return the name
         */
        public String getName() {
            return name;
        }

        /**
         * @return the variable
         */
        public Variable getVariable() {
            return variable;
        }

        /**
         * @return the initializer
         */
        public Initializer getInitializer() {
            return initializer;
        }
        private final String name;
        private final Variable variable;
        private final Initializer initializer;
        
        public Slot(String name, Variable variable, Initializer initializer) {
            this.name =name;
            this.variable = variable;
            this.initializer = initializer;
        }
             
    }

}
