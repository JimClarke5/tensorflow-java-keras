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
import org.tensorflow.framework.optimizers.Optimizer;
import static org.tensorflow.framework.optimizers.RMSProp.MG;
import static org.tensorflow.framework.optimizers.RMSProp.MOMENTUM;
import static org.tensorflow.framework.optimizers.RMSProp.RMS;
import static org.tensorflow.keras.optimizers.Ftrl.LEARNING_RATE_KEY;
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;
import static org.tensorflow.keras.optimizers.RMSProp.CENTERED_DEFAULT;
import static org.tensorflow.keras.optimizers.RMSProp.CENTERED_KEY;
import static org.tensorflow.keras.optimizers.RMSProp.DECAY_DEFAULT;
import static org.tensorflow.keras.optimizers.RMSProp.DECAY_KEY;
import static org.tensorflow.keras.optimizers.RMSProp.EPSILON_DEFAULT;
import static org.tensorflow.keras.optimizers.RMSProp.EPSILON_KEY;
import static org.tensorflow.keras.optimizers.RMSProp.MOMENTUM_DEFAULT;
import static org.tensorflow.keras.optimizers.RMSProp.MOMENTUM_KEY;
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
public class RMSPropTest {
    
    final int VAR_T = 0;
    final int MG_T= 1;
    final int RMS_T= 2;
    final int MOM_T = 3;

    int index;

    public RMSPropTest() {
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
     * Test of create method, of class RMSProp.
     */
    @Test
    public void testCreate() {
        System.out.println("create");
        try ( Graph graph = new Graph()) {
            Map<String, Object> config = new HashMap<>();
            config.put(NAME_KEY, "Ftrl");
            config.put(LEARNING_RATE_KEY, 2.0F);
            config.put(DECAY_KEY, DECAY_DEFAULT);
            config.put(MOMENTUM_KEY, MOMENTUM_DEFAULT);
            config.put(EPSILON_KEY, EPSILON_DEFAULT);
            config.put(CENTERED_KEY, CENTERED_DEFAULT);
            Ftrl expResult = new Ftrl(graph, 2.0F);
            Ftrl result = Ftrl.create(graph, config);
            assertEquals(expResult.getConfig(), result.getConfig());
        }
    }

    Object[][] _test_param_values = {
        // learning_rate, rho (decay), momentum, epsilon, centered
        {0.05F, 0.9F, 0.0F, 1e-3F, true},
        {0.05F, 0.9F, 0.0F, 1e-3F, false},
        {0.1F, 0.9F, 0.0F, 1e-3F, true},
        {0.01F, 0.9F, 0.0F, 1e-5F, true},
        {0.01F, 0.9F, 0.9F, 1e-5F, true}
    };

    @Test
    public void testDense() {
        System.out.println("testDense");

        int numSteps = 3;

        for (int run = 0; run < _test_param_values.length; run++) {
            try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                Ops tf = Ops.create(graph).withName("test");
                float[] var0_init = {1.0F, 2.0F};
                float[] var1_init = {3.0F, 4.0F};
                float[] grads0_init = {0.1F, 0.2F};
                float[] grads1_init = {0.01F, 0.2F};
                final float epsilon1 = 1e-2F;

                FloatNdArray var0_np = NdArrays.vectorOf(var0_init);
                FloatNdArray var1_np = NdArrays.vectorOf(var1_init);
                FloatNdArray grads0_np = NdArrays.vectorOf(grads0_init);
                FloatNdArray grads1_np = NdArrays.vectorOf(grads1_init);

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

                // learning_rate, rho (decay), momentum, epsilon, centered
                float learningRate = (float) (float) _test_param_values[run][0];
                float decay = (float) _test_param_values[run][1];
                float momentum = (float) _test_param_values[run][2];
                float epsilon = (float) _test_param_values[run][3];
                boolean centered = (boolean) _test_param_values[run][4];
                
                System.out.printf("\nRMSProp: learningRate=%f, decay=%f, momentum=%f, epsilon=%f, centered=%s\n",
                        learningRate, decay, momentum, epsilon, centered);

                RMSProp instance = new RMSProp(graph,
                        learningRate,
                        decay,
                        momentum,
                        epsilon,
                        centered);

                /* build the GradsAnvVars */
                List gradsAndVars = new ArrayList<>();
                gradsAndVars.add(new Optimizer.GradAndVar<>(grads0.asOutput(), var0.asOutput()));
                gradsAndVars.add(new Optimizer.GradAndVar<>(grads1.asOutput(), var1.asOutput()));

                Op update = instance.applyGradients(gradsAndVars, "RMSPropTest");

                /* initialize the local variables */
                sess.runner().addTarget(var0Initializer).run();
                sess.runner().addTarget(var1Initializer).run();

                /**
                 * initialize the accumulators
                 */
                graph.initializers().forEach((initializer) -> {
                    sess.runner().addTarget(initializer).run();
                });
                
                

                Variable<TFloat32> mg0 = centered ? instance.getSlot(var0.asOutput(), MG).get() : null;
                Variable<TFloat32> mg1 = centered ? instance.getSlot(var1.asOutput(), MG).get() : null;
                Variable<TFloat32> mom0 = momentum > 0.F ? instance.getSlot(var0.asOutput(), MOMENTUM).get() : null;
                Variable<TFloat32> mom1 = momentum > 0.F ? instance.getSlot(var1.asOutput(), MOMENTUM).get() : null;
                Variable<TFloat32> rms0 = instance.getSlot(var0.asOutput(), RMS).get();
                Variable<TFloat32> rms1 = instance.getSlot(var1.asOutput(), RMS).get();

                float[] zeros = {0.0F, 0.0F};
                float[] ones = {1.0F, 1.0F}; // temp to match RMSProp
                FloatNdArray mg0_np = NdArrays.vectorOf(zeros);
                FloatNdArray mg1_np = NdArrays.vectorOf(zeros);
                FloatNdArray rms0_np = NdArrays.vectorOf(ones);
                FloatNdArray rms1_np = NdArrays.vectorOf(ones);
                FloatNdArray mom0_np = NdArrays.vectorOf(zeros);
                FloatNdArray mom1_np = NdArrays.vectorOf(zeros);

                try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    result.data().scalars().forEach(f -> assertEquals(var0_init[index++], f.getFloat(), epsilon));
                }
                try ( Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                    index = 0;
                    result.data().scalars().forEach(f -> assertEquals(var1_init[index++], f.getFloat(), epsilon));
                }

                for (int i = 0; i < numSteps; i++) {
                    sess.run(update);
                    System.out.println("Step: " + i+1);
                    System.out.println("================== var0 ======================================");
                    FloatNdArray[] result0 = calc(var0_np, grads0_np, mg0_np, rms0_np,
                            mom0_np, learningRate, decay, momentum, epsilon, centered);
                    var0_np = result0[VAR_T];
                    mg0_np = result0[MG_T];
                    rms0_np = result0[RMS_T];
                    mom0_np = result0[MOM_T];

                    System.out.println("================== var01 ======================================");
                    FloatNdArray[] result1 = calc(var1_np, grads1_np, mg1_np, rms1_np,
                            mom1_np, learningRate, decay, momentum, epsilon, centered);

                    var1_np = result1[VAR_T];
                    mg1_np = result1[MG_T];
                    rms1_np = result1[RMS_T];
                    mom1_np = result1[MOM_T];

                    if (centered) {
                        try ( Tensor<TFloat32> result = sess.runner().fetch(mg0).run().get(0).expect(TFloat32.DTYPE)) {
                            index = 0;
                            final FloatNdArray ftmp = mg0_np;
                            result.data().scalars().forEach(f -> {
                                System.out.printf("mg0_np: %f, mg0: %f\n",ftmp.getFloat(index), f.getFloat());
                                assertEquals(ftmp.getFloat(index++), f.getFloat(), epsilon1);
                                    });
                        }
                        try ( Tensor<TFloat32> result = sess.runner().fetch(mg1).run().get(0).expect(TFloat32.DTYPE)) {
                            index = 0;
                            final FloatNdArray ftmp = mg1_np;
                            result.data().scalars().forEach(f -> {
                                System.out.printf("mg1_np: %f, mg1: %f\n",ftmp.getFloat(index), f.getFloat());
                                assertEquals(ftmp.getFloat(index++), f.getFloat(), epsilon1);
                             });
                        }
                    }
                    if (momentum > 0.F) {
                        try ( Tensor<TFloat32> result = sess.runner().fetch(mom0).run().get(0).expect(TFloat32.DTYPE)) {
                            index = 0;
                            final FloatNdArray ftmp = mom0_np;
                            result.data().scalars().forEach(f -> {
                               System.out.printf("mom0_np: %f, mom0: %f\n",ftmp.getFloat(index), f.getFloat());
                                assertEquals(ftmp.getFloat(index++), f.getFloat(), epsilon1);
                                    });
                        }
                        try ( Tensor<TFloat32> result = sess.runner().fetch(mom1).run().get(0).expect(TFloat32.DTYPE)) {
                            index = 0;
                            final FloatNdArray ftmp = mom1_np;
                            result.data().scalars().forEach(f -> {
                               System.out.printf("mom1_np: %f, mom1: %f\n",ftmp.getFloat(index), f.getFloat());
                                assertEquals(ftmp.getFloat(index++), f.getFloat(), epsilon1);
                             });
                        }
                    }

                    /*     TODO the values returned from rms slot, do not match what I see in the python test */
                    try ( Tensor<TFloat32> result = sess.runner().fetch(rms0).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        final FloatNdArray ftmp = rms0_np;
                        result.data().scalars().forEach(
                                f -> {
                                    System.out.printf("rms0_np: %f, rms0: %f\n",ftmp.getFloat(index), f.getFloat());
                                    assertEquals(ftmp.getFloat(index++), f.getFloat(), epsilon1);
                                }
                        );
                    }
                    
                    try ( Tensor<TFloat32> result = sess.runner().fetch(rms1).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        final FloatNdArray ftmp = rms1_np;
                        
                        result.data().scalars().forEach(
                                f -> {
                                    System.out.printf("rms1_np: %f, rms1: %f\n",ftmp.getFloat(index), f.getFloat());
                                    assertEquals(ftmp.getFloat(index++), f.getFloat(), epsilon1);
                                }
                        );                    
                    }
                    try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        final FloatNdArray ftmp = var0_np;
                        result.data().scalars().forEach(f -> {
                            System.out.printf("var0_np: %f, var0: %f\n",ftmp.getFloat(index), f.getFloat());
                            assertEquals(ftmp.getFloat(index++), f.getFloat(), epsilon1);
                         });
                    }
                    

                    try ( Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        final FloatNdArray ftmp = var1_np;
                        result.data().scalars().forEach(f -> {
                            System.out.printf("var1_np: %f, var1: %f\n",ftmp.getFloat(index), f.getFloat());
                            assertEquals(ftmp.getFloat(index++), f.getFloat(), epsilon1);
                         });
                    }
                    

                }

            }
        }
    }

    FloatNdArray[] calc(FloatNdArray var_np, FloatNdArray grad_np, FloatNdArray mg_np,
            FloatNdArray rms_np, FloatNdArray mom, float lr, float decay, float momentum,
            float epsilon, boolean centered) {

        FloatNdArray[] result = new FloatNdArray[4]; // var_t, mg_t, rms_t, mom_t
        result[RMS_T] = calcRMS(rms_np, grad_np, decay); // RMS

        FloatNdArray denom_t;
        if (centered) {
            result[MG_T] = calcMG(mg_np, grad_np, decay);
            //rms_t - mg_t * mg_t
            denom_t = NP.sub(result[RMS_T], NP.square(result[MG_T]));
        } else {
            result[MG_T] = mg_np;
            denom_t = rms_np;
        }
        if (momentum > 0.F) {
            //momentum * mom + lr * g / (np.sqrt(denom_t + epsilon))
            result[MOM_T] = calcMom(momentum, mom, lr, grad_np, denom_t, epsilon);
            //var_t = var - mom_t
            result[VAR_T] = NP.sub(var_np, result[MOM_T]);
        } else {
            result[MOM_T] = mom;
            result[VAR_T] = calcVar(var_np, grad_np, lr, denom_t, epsilon);
        }
        NP.print("var_t", result[VAR_T]);
        NP.print("mg_t",result[MG_T]);
        NP.print("rms_t",result[RMS_T]);
        NP.print("mom_t",result[MOM_T]);

        return result;

    }

    private FloatNdArray calcRMS(FloatNdArray rms_np, FloatNdArray grad_np, float decay) {
        //rms * rho + (1 - rho) * g * g
        FloatNdArray rms_rho = NP.mul(rms_np, decay);
        FloatNdArray squareG = NP.square(grad_np);
        float oneRHO = 1.0F - decay;
        FloatNdArray decayG2 = NP.mul(oneRHO, squareG);
        FloatNdArray result = NP.add(rms_rho, decayG2);
        return result;
    }

    private FloatNdArray calcMG(FloatNdArray mg_np, FloatNdArray grad_np, float decay) {
        //mg_t = mg * rho + (1 - rho) * g
        FloatNdArray mg_rho = NP.mul(mg_np, decay);
        float oneRHO = 1.0F - decay;
        FloatNdArray decayG = NP.mul(oneRHO, grad_np);
        FloatNdArray result = NP.add(mg_rho, decayG);
        return result;

    }

    private FloatNdArray calcMom(float momentum, FloatNdArray mom, float lr,
            FloatNdArray grad_np, FloatNdArray denom_t, float epsilon) {
        // momentum * mom + lr * g / (np.sqrt(denom_t + epsilon))
        FloatNdArray moMo = NP.mul(momentum, mom);
        FloatNdArray dividend = NP.mul(lr, grad_np);
        FloatNdArray divisor = NP.sqrt(NP.add(denom_t, epsilon));
        FloatNdArray quotient = NP.div(dividend, divisor);
        FloatNdArray result = NP.add(moMo, quotient);
        return result;

    }

    private FloatNdArray calcVar(FloatNdArray var_np, FloatNdArray grad_np, float lr,
            FloatNdArray denom_t, float epsilon) {
        // var - lr * g / (np.sqrt(denom_t) + epsilon)
        FloatNdArray dividend = NP.mul(lr, grad_np);
        FloatNdArray divisor = NP.add(NP.sqrt(denom_t), epsilon);
        FloatNdArray quotient = NP.div(dividend, divisor);
        FloatNdArray result = NP.sub(var_np, quotient);
        return result;

    }
}
