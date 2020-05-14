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
import static org.tensorflow.framework.optimizers.Adam.FIRST_MOMENT;
import static org.tensorflow.framework.optimizers.Adam.SECOND_MOMENT;
import org.tensorflow.framework.optimizers.Optimizer;
import static org.tensorflow.keras.optimizers.Adam.BETA_ONE_DEFAULT;
import static org.tensorflow.keras.optimizers.Adam.BETA_ONE_KEY;
import static org.tensorflow.keras.optimizers.Adam.BETA_TWO_DEFAULT;
import static org.tensorflow.keras.optimizers.Adam.BETA_TWO_KEY;
import static org.tensorflow.keras.optimizers.Adam.EPSILON_DEFAULT;
import static org.tensorflow.keras.optimizers.Adam.EPSILON_KEY;
import static org.tensorflow.keras.optimizers.Adam.LEARNING_RATE_DEFAULT;
import static org.tensorflow.keras.optimizers.Adam.LEARNING_RATE_KEY;
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;
import org.tensorflow.keras.utils.NdHelper;
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
public class AdamTest {
    
    int index;
    
    public AdamTest() {
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
     * Test of create method, of class Adam.
     */
    @Test
    public void testCreate() {
        System.out.println("create");
        try ( Graph graph = new Graph()) {
            Map<String, Object> config = new HashMap<>();
            config.put(NAME_KEY, "AdaDelta");
            config.put(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
            config.put(BETA_ONE_KEY, BETA_ONE_DEFAULT);
            config.put(BETA_TWO_KEY, BETA_TWO_DEFAULT);
            config.put(EPSILON_KEY, EPSILON_DEFAULT);
            AdaDelta expResult = new AdaDelta(graph);
            AdaDelta result = AdaDelta.create(graph, config);
            assertEquals(expResult.getConfig(), result.getConfig());
        }
    }

    @Test
    public void testBasic() {
        System.out.println("testBasic");
        float m0 = 0.0F;
        float v0 = 0.0F;
        float m1 = 0.0F;
        float v1 = 0.0F;
        float[] var0_init = {1.0F, 2.0F};
        float[] var1_init = {3.0F, 4.0F};
        float[] grads0_init = {0.1F, 0.1F};
        float[] grads1_init = {0.01F, 0.01F};
        FloatNdArray var0_np = NdArrays.vectorOf(var0_init);
        FloatNdArray var1_np = NdArrays.vectorOf(var1_init);
        FloatNdArray grads0_np = NdArrays.vectorOf(grads0_init);
        FloatNdArray grads1_np = NdArrays.vectorOf(grads1_init);
        
        
        float epsilon1 = 1e-2F;
        
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
            
            float learningRate = 0.001F;
            float beta1 = 0.9F;
            float beta2 = 0.999F;
            float epsilon = 1e-8F;
            
            /* build the GradsAnvVars */
            List gradsAndVars = new ArrayList<>();
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads0.asOutput(), var0.asOutput()));
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads1.asOutput(), var1.asOutput()));
            
            Adam instance = new Adam(graph, learningRate);
            

            Op update = instance.applyGradients(gradsAndVars, "AdamTest");
            
            /* Create and validae the shapes of the slota */
            Variable<TFloat32>[] firstMomentSlots = new Variable[2];
            Variable<TFloat32>[] secondMomentSlots = new Variable[2];

            firstMomentSlots[0] = instance.getSlot(var0.asOutput(), FIRST_MOMENT).get();
            assertEquals(firstMomentSlots[0].asOutput().shape(), var0.asOutput().shape());

            secondMomentSlots[0] = instance.getSlot(var0.asOutput(), SECOND_MOMENT).get();
            assertEquals(secondMomentSlots[0].asOutput().shape(), var0.asOutput().shape());

            firstMomentSlots[1] = instance.getSlot(var1.asOutput(), FIRST_MOMENT).get();
            assertEquals(firstMomentSlots[1].asOutput().shape(), var1.asOutput().shape());

            secondMomentSlots[1] = instance.getSlot(var1.asOutput(), SECOND_MOMENT).get();
            assertEquals(secondMomentSlots[1].asOutput().shape(), var1.asOutput().shape());
            
            /**
            * initialize the accumulators
            */
            
            for(Op initializer : graph.initializers()) {
                sess.runner().addTarget(initializer).run();
            }
            
            try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> assertEquals(var0_init[index++], f.getFloat(), epsilon));
            }
            try ( Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> assertEquals(var1_init[index++], f.getFloat(), epsilon));
            }
            
            FloatNdArray m0_np = NdArrays.ofFloats(shape1);
            FloatNdArray v0_np = NdArrays.ofFloats(shape1);
            FloatNdArray m1_np = NdArrays.ofFloats(shape1);
            FloatNdArray v1_np = NdArrays.ofFloats(shape1);
                    
            for(int step =0; step < 3; step++) {
                
                // Test powers
                final float[] powers = {(float)Math.pow(beta1, step+1), (float)Math.pow(beta2,  step+1)};
                
                try ( Tensor<TFloat32> result = sess.runner().fetch("beta1_power").run().get(0).expect(TFloat32.DTYPE)) {
                    result.data().scalars().forEach(f -> 
                    {
                        assertEquals(powers[0], f.getFloat(), epsilon1);
                    });
                }
                try ( Tensor<TFloat32> result = sess.runner().fetch("beta2_power").run().get(0).expect(TFloat32.DTYPE)) {
                    result.data().scalars().forEach(f -> 
                    {
                        assertEquals(powers[1], f.getFloat(), epsilon1);
                    });
                }
                sess.run(update);
                
                float lr_t = learningRate * (float)Math.sqrt(1 - (float)Math.pow(beta2, (step + 1))) / (1 - (float)Math.pow(beta1, (step + 1)));
                
                m0_np = calculateM(m0_np, grads0_np, beta1);
                v0_np = calculateV(v0_np, grads0_np, beta2);
                var0_np = calculateParam(var0_np, lr_t, m0_np, v0_np, 1e-7F);
                
                m1_np = calculateM(m1_np, grads1_np, beta1);
                v1_np = calculateV(v1_np, grads1_np, beta2);
                var1_np = calculateParam(var1_np, lr_t, m1_np, v1_np, 1e-7F);
                
                // get var0
                try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    final FloatNdArray var0_final = var0_np;
                    result.data().scalars().forEach(f-> {
                        assertEquals(var0_final.getFloat(index++), f.getFloat(), epsilon1);
                    } );
                }


                // get var1
                
                try ( Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    final FloatNdArray var1_final = var1_np;
                    result.data().scalars().forEach(f-> {
                        assertEquals(var1_final.getFloat(index++), f.getFloat(), epsilon1);
                    } );
                }
                
                // first moment
                try ( Tensor<TFloat32> result = sess.runner().fetch(firstMomentSlots[0]).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    final FloatNdArray firstMomentSlot0_final = m0_np;
                    result.data().scalars().forEach(f-> {
                        assertEquals(firstMomentSlot0_final.getFloat(index++), f.getFloat(), epsilon1);
                    } );
                }
                try ( Tensor<TFloat32> result = sess.runner().fetch(firstMomentSlots[1]).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    final FloatNdArray firstMomentSlot1_final = m1_np;
                    result.data().scalars().forEach(f-> {
                        assertEquals(firstMomentSlot1_final.getFloat(index++), f.getFloat(), epsilon1);
                    } );
                }
                
                // second moment
                try ( Tensor<TFloat32> result = sess.runner().fetch(secondMomentSlots[0]).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    final FloatNdArray secondMomentSlot0_final = v0_np;
                    result.data().scalars().forEach(f-> {
                        assertEquals(secondMomentSlot0_final.getFloat(index++), f.getFloat(), epsilon1);
                    } );
                }
                try ( Tensor<TFloat32> result = sess.runner().fetch(secondMomentSlots[1]).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    final FloatNdArray secondMomentSlot1_final = v1_np;
                    result.data().scalars().forEach(f-> {
                        assertEquals(secondMomentSlot1_final.getFloat(index++), f.getFloat(), epsilon1);
                    } );
                }
                
                
            }

        }
        
    }
    
    private FloatNdArray calculateM(FloatNdArray m, FloatNdArray g_t, float beta) {
        // m_t = beta1 * m + (1 - beta1) * g_t
        return NdHelper.add(NdHelper.mul(m, beta), NdHelper.mul(g_t, (1-beta)));
    }
    
    private FloatNdArray calculateV(FloatNdArray v, FloatNdArray g_t, float beta) {
        //beta2 * v + (1 - beta2) * g_t * g_t
        FloatNdArray mul1 = NdHelper.mul(v, beta);
        FloatNdArray sqr = NdHelper.squared(g_t);
        FloatNdArray mul2 = NdHelper.mul((1-beta), sqr);
        FloatNdArray add = NdHelper.add(mul1, mul2);
        return add;
        
        //return NdHelper.add(NdHelper.mul(v, beta),
        //     NdHelper.mul((1-beta), NdHelper.squared(g_t)));
    }
    
    private FloatNdArray calculateParam(FloatNdArray param, float lr_t, FloatNdArray m,  FloatNdArray v, float epsilon) {
       //  param - lr_t * m_t / (np.sqrt(v_t) + epsilon)
       FloatNdArray sqrt = NdHelper.sqrt(v);
       FloatNdArray divisor = NdHelper.add(sqrt, epsilon);
       FloatNdArray dividend = NdHelper.mul(lr_t, m);
       FloatNdArray quotient = NdHelper.div(dividend, divisor);
       FloatNdArray result = NdHelper.minus(param, quotient);
       return result;
       
    }
    
    
    
}
