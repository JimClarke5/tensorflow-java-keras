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
public class BinaryCrossentropyTest {

    int index;
    float epsilon = 1e-4F;

    public BinaryCrossentropyTest() {
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
        BinaryCrossentropy instance = new BinaryCrossentropy();
        assertEquals("binary_crossentropy", instance.getName());

        instance = new BinaryCrossentropy("bce_1", Reduction.SUM);
        assertEquals("bce_1", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void testAllCorrectUnweighted() {
        System.out.println("testAllCorrectUnweighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            BinaryCrossentropy instance = new BinaryCrossentropy();
            float[] true_np = {1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            
            Operand loss = instance.call(tf, y_true, y_true);
            
            sess.run(loss);
            float expected = 0.0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }
            System.out.println("============ LOGITS =================");
            // Test with logits.
            float[] logits_np = {
                100.0F, -100.0F, -100.0F,
                -100.0F, 100.0F, -100.0F,
                -100.0F, -100.0F, 100.0f
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new BinaryCrossentropy(true);
            
            loss = instance.call(tf, y_true, logits);
            sess.run(loss);
            float expected1 = 0.0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected1, f.getFloat(), epsilon);
                });
            }
        }
    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            BinaryCrossentropy instance = new BinaryCrossentropy();
            float[] true_np = {1F, 0F, 1F, 0F};
            float[] pred_np = {1F, 1F, 1F, 0F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = instance.call(tf, y_true, y_pred);
            sess.run(loss);
            float expected = 3.83331F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }

            // Test with logits.
            float[] true_np1 = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            instance = new BinaryCrossentropy(true);
            loss = instance.call(tf, y_true1, logits);
            sess.run(loss);
            float expected1 = 33.33333F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected1, f.getFloat(), epsilon);
                });
            }
        }
    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");
            BinaryCrossentropy instance = new BinaryCrossentropy();
            float[] true_np = {1F, 0F, 1F, 0F};
            float[] pred_np = {1F, 1F, 1F, 0F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 8.8166F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }

            // Test with logits.
            float[] true_np1 = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            instance = new BinaryCrossentropy(true);
            loss = instance.call(tf, y_true1, logits, sampleWeight);
            sess.run(loss);
            float expected1 = 76.66667F;
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
            BinaryCrossentropy instance = new BinaryCrossentropy();
            float[] true_np = {1F, 0F, 1F, 0F};
            float[] pred_np = {1F, 1F, 1F, 0F};
            float[] sample_weight_np = {1.2F, 3.4F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_weight_np), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(tf, y_true, y_pred, sampleWeight);
            sess.run(loss);
            float expected = 4.59997F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }

            // Test with logits.
            float[] true_np1 = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            int[] weights_np = {4, 3};
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight1 = tf.constant(weights_np);
            instance = new BinaryCrossentropy(true);
            loss = instance.call(tf, y_true1, logits, sampleWeight1);
            sess.run(loss);
            float expected1 = 100F;
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
            float[] true_np1 = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            BinaryCrossentropy instance = new BinaryCrossentropy(true, 0.0F, Reduction.NONE);
            Operand loss = instance.call(tf, y_true1, logits);
            sess.run(loss);
            float[] expected = {0.F, 66.6666F};
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
            float[] true_array = {1f, 0f, 1f};
            float[] logits_array = {100.0f, -100.0f, -100.0f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(1, 3)));
            Operand logits = tf.reshape(tf.constant(logits_array), tf.constant(Shape.of(1, 3)));
            
            BinaryCrossentropy instance = new BinaryCrossentropy(true, label_smoothing);
            Operand loss = instance.call(tf, y_true, logits);
            System.out.println(loss.asOutput().shape());
            sess.run(loss);
            float expected =  (100.0F + 50.0F * label_smoothing) / 3.0F;
            try ( Tensor<TFloat32> result = sess.runner().fetch(loss).run().get(0).expect(TFloat32.DTYPE)) {
                result.data().scalars().forEach(f -> {
                    assertEquals(expected, f.getFloat(), epsilon);
                });
            }
        } catch (Exception expected) {

        }
    }

}
