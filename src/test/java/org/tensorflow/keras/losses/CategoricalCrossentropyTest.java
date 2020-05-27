/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class CategoricalCrossentropyTest {
    int index;
    float epsilon = 1e-4F;
    
    public CategoricalCrossentropyTest() {
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

    @Test
    public void testConfig() {
        System.out.println("testConfig");
        CategoricalCrossentropy instance = new CategoricalCrossentropy(null);
        assertEquals("categorical_crossentropy", instance.getName());

        instance = new CategoricalCrossentropy(null, "catx_1", Reduction.SUM);
        assertEquals("catx_1", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }
    

    /**
     * Test of call method, of class CategoricalCrossentropy.
     */
    @Test
    public void testAllCorrectUnweighted() {
        System.out.println("testAllCorrectUnweighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            
            long[] true_np = {
                1L, 0L, 0L,
                0L, 1L, 0L,
                0L, 0L, 1L};
            float[] pred_np = { 
                1.F, 0.F, 0.F,
                0.F, 1.F, 0.F,
                0.F, 0.F, 1.F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            Operand loss = instance.call(y_true, y_pred);
            sess.run(loss);
            float expected = 0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }
            
            System.out.println("============ LOGITS =================");
            // Test with logits.
            float[] logits_np = {
                10.F, 0.F, 0.F,
                0.F, 10.F, 0.F,
                0.F, 0.F, 10.F
            };
            y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new CategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits);
            sess.run(loss);
            float expected1 = 9.083335E-5F; // 9.083335E-4, but should be 9.083335E-5
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected1, f.getFloat(), epsilon);
                });
            }
        }
    }

    /**
     * Test of call method, of class CategoricalCrossentropy.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            int[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = instance.call(y_true, y_pred);
            sess.run(loss);
            float expected = .3239F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new CategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits);
            sess.run(loss);
            float expected1 = .0573F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected1, f.getFloat(), epsilon);
                });
            }
        }
    }

    /**
     * Test of call method, of class CategoricalCrossentropy.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            
            int[] true_np = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand sampleWeight = tf.constant(2.3f);
            
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = .7451267F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new CategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits, sampleWeight);
            sess.run(loss);
            float expected1 = 0.13196386F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected1, f.getFloat(), epsilon);
                });
            }
        }
    }

    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            float[] sample_weight_np = {1.2F, 3.4F, 5.6F};
            int[] true_np = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_weight_np), tf.constant(Shape.of(3, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 1.0696F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new CategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits, sampleWeight);
            sess.run(loss);
            float expected1 = 0.31829F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected1, f.getFloat(), epsilon);
                });
            }
        }
    }

    @Test
    public void test_no_reduction() {
        System.out.println("test_no_reduction");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");

            // Test with logits.
            int[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, true, 0.0F, Reduction.NONE);
            Operand loss = instance.call(y_true, logits);
            sess.run(loss);
            float[] expected = {0.001822F, 0.000459F, 0.169846F};
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> {
                    assertEquals(expected[index++], f.getFloat(), epsilon);
                });
            }
        }
    }

    @Test
    public void test_label_smoothing() {
        System.out.println("test_label_smoothing");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            float label_smoothing = 0.1f;
            int[] true_array = {1, 0, 0};
            float[] logits_array = {100.0f, -100.0f, -100.0f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(1, 3)));
            Operand logits = tf.reshape(tf.constant(logits_array), tf.constant(Shape.of(1, 3)));
            
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, true, label_smoothing);
            Operand loss = instance.call(y_true, logits);
            System.out.println(loss.asOutput().shape());
            sess.run(loss);
            float expected =  400.0F * label_smoothing / 3.0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }
        } catch (Exception expected) {

        }
    }
    
}
