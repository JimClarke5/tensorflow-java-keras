/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.optimizers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import static org.tensorflow.framework.optimizers.AdaGrad.ACCUMULATOR;
import org.tensorflow.framework.optimizers.Optimizer;
import static org.tensorflow.keras.optimizers.AdaGrad.INITIAL_ACCUM_KEY;
import static org.tensorflow.keras.optimizers.AdaGrad.LEARNING_RATE_KEY;
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;
import org.tensorflow.keras.utils.NP;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.FloatNdArray;
import org.tensorflow.tools.ndarray.NdArrays;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class AdaGradTest {
    
    int index;
    
    public AdaGradTest() {
    }
    
    @BeforeAll
    public static void setUpClass() {
    }
    
    @AfterAll
    public static void tearDownClass() {
    }
    
    @BeforeEach
    public void setUp() {
    }
    
    @AfterEach
    public void tearDown() {
    }


    /**
     * Test of create method, of class AdaGrad.
     */
    @Test
    public void testCreate() {
        System.out.println("create");
         try ( Graph graph = new Graph()) {
            Map<String, Object> config = new HashMap<>();
            config.put(NAME_KEY, "AdaDelta");
            config.put(LEARNING_RATE_KEY, 2.0F);
            config.put(INITIAL_ACCUM_KEY, 0.1F);
            AdaGrad expResult = new AdaGrad(graph, 2.0F, 0.1F);
            AdaGrad result = AdaGrad.create(graph, config);
            assertEquals(expResult.getConfig(), result.getConfig());
        }
    }
    
    @Test
    public void testBasic() {
        System.out.println("testBasic");
        int numSteps = 3;
        float[] var0_init = {1.0F, 2.0F};
        float[] var1_init = {3.0F, 4.0F};
        float[] grads0_init = {0.1F, 0.1F};
        float[] grads1_init = {0.01F, 0.01F};
        float epsilon = 1e-8F;
        float epsilon1 = 1e-5F;
        float[] accum0 = { 0.1f, 0.1f};
        float[] accum1 = { 0.1f, 0.1f};
        
        
        FloatNdArray var0_np = NdArrays.vectorOf(var0_init);
        FloatNdArray var1_np = NdArrays.vectorOf(var1_init);
        FloatNdArray grads0_np = NdArrays.vectorOf(grads0_init);
        FloatNdArray grads1_np = NdArrays.vectorOf(grads1_init);
        FloatNdArray accum0_np = NdArrays.vectorOf(accum0);
        FloatNdArray accum1_np = NdArrays.vectorOf(accum1);
        
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            
            Shape shape0 = Shape.of(var0_init.length);
            Shape shape1 = Shape.of(var1_init.length);
            Variable<TFloat32> var0 = tf.withName("var0").variable(shape0, TFloat32.DTYPE);
            Variable<TFloat32> var1 = tf.withName("var1").variable(shape1, TFloat32.DTYPE);
            
            Assign<TFloat32> var0Initializer = tf.assign(var0, tf.constant(var0_init));
            Assign<TFloat32> var1Initializer = tf.assign(var1, tf.constant(var1_init));
            
            Constant<TFloat32> grads0 = tf.constant(grads0_init);
            Constant<TFloat32> grads1 = tf.constant(grads1_init);
            
             /* initialize the local variables */
            sess.runner().addTarget(var0Initializer).run();
            sess.runner().addTarget(var1Initializer).run();
            
            float learningRate = 3.0F;
            
            AdaGrad instance = new AdaGrad(graph, learningRate);
            

            
            /* build the GradsAnvVars */
            List gradsAndVars = new ArrayList<>();
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads0.asOutput(), var0.asOutput()));
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads1.asOutput(), var1.asOutput()));
            
            Op ada_update = instance.applyGradients(gradsAndVars, "AdGradTest");
            
            Variable<TFloat32>[] accumulatorSlots = new Variable[2];
            accumulatorSlots[0] = instance.getSlot(var0.asOutput(), ACCUMULATOR).get();
            assertEquals(accumulatorSlots[0].asOutput().shape(), var0.asOutput().shape());
            
            accumulatorSlots[1] = instance.getSlot(var1.asOutput(), ACCUMULATOR).get();
            assertEquals(accumulatorSlots[1].asOutput().shape(), var1.asOutput().shape());
            
            /* initialize the local variables */
            sess.runner().addTarget(var0Initializer).run();
            sess.runner().addTarget(var1Initializer).run();


            /**
             * initialize the accumulators
             */
            graph.initializers().forEach((initializer) -> {
                sess.runner().addTarget(initializer).run();
            });
            
            try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> assertEquals(var0_init[index++], f.getFloat(), epsilon));
            }
            try ( Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> assertEquals(var1_init[index++], f.getFloat(), epsilon));
            }
            
            for(int step = 0; step < numSteps; step++) {
                sess.run(ada_update);
                
                accum0_np = caclulateAccum(accum0_np, grads0_np);
                var0_np = calculate(var0_np, accum0_np, grads0_np, learningRate);
                try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    final FloatNdArray var0_final = var0_np;
                    result.data().scalars().forEach(f -> assertEquals(var0_final.getFloat(index++), f.getFloat(), epsilon1));
                }
                
                accum1_np =  caclulateAccum(accum1_np, grads1_np);
                var1_np = calculate(var1_np, accum1_np, grads1_np, learningRate);
                try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    final FloatNdArray var0_final = var0_np;
                    result.data().scalars().forEach(f -> assertEquals(var0_final.getFloat(index++), f.getFloat(), epsilon1));
                }
            }
            
            
        }
    }
    
    private FloatNdArray caclulateAccum(FloatNdArray accum, FloatNdArray grads) {
        // accum + g_t * g_t
        FloatNdArray squareG = NP.square(grads);
        FloatNdArray result = NP.add(accum, squareG);
        NP.print(result);
        return result;
    }

    private FloatNdArray calculate(FloatNdArray param, FloatNdArray accum, FloatNdArray grads, float learningRate) {
        //param - lr * g_t / (np.sqrt(accum_t) + epsilon)
        FloatNdArray divisor = NP.add(NP.sqrt(accum), 1e-07f);
        FloatNdArray dividend = NP.mul(learningRate, grads);
        FloatNdArray quotient = NP.div(dividend, divisor);
        FloatNdArray result = NP.sub(param, quotient);
        return result;
    }
    
}
