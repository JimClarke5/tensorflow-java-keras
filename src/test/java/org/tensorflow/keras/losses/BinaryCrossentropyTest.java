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
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.keras.utils.TestSession.Mode;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class BinaryCrossentropyTest {

    private Mode tf_mode = Mode.EAGER;

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
        BinaryCrossentropy instance = new BinaryCrossentropy(null);
        assertEquals("binary_crossentropy", instance.getName());

        instance = new BinaryCrossentropy(null, "bce_1", Reduction.SUM);
        assertEquals("bce_1", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void testAllCorrectUnweighted() {
        System.out.println("testAllCorrectUnweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            float[] true_np = {1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));

            Operand<TFloat32> loss = instance.call(y_true, y_true);

            float expected = 0.0F;
            testSession.evaluate(expected, loss);
            System.out.println("============ LOGITS =================");
            // Test with logits.
            float[] logits_np = {
                100.0F, -100.0F, -100.0F,
                -100.0F, 100.0F, -100.0F,
                -100.0F, -100.0F, 100.0f
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new BinaryCrossentropy(tf, true);

            loss = instance.call(y_true, logits);
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            float[] true_np = {1F, 0F, 1F, 0F};
            float[] pred_np = {1F, 1F, 1F, 0F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 3.83331F;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] true_np1 = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            instance = new BinaryCrossentropy(tf, true);
            loss = instance.call(y_true1, logits);
            expected = 33.33333F;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void test_scalar_weighted() {
        System.out.println("test_scalar_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            float[] true_np = {1F, 0F, 1F, 0F};
            float[] pred_np = {1F, 1F, 1F, 0F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 8.816612F;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] true_np1 = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            instance = new BinaryCrossentropy(tf, true);
            loss = instance.call(y_true1, logits, sampleWeight);
            expected = 76.66667F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        System.out.println("test_sample_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            float[] true_np = {1F, 0F, 1F, 0F};
            float[] pred_np = {1F, 1F, 1F, 0F};
            float[] sample_weight_np = {1.2F, 3.4F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_weight_np), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 4.59997F;
            testSession.evaluate(expected, loss);

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
            instance = new BinaryCrossentropy(tf, true);
            loss = instance.call(y_true1, logits, sampleWeight1);
            expected = 100F;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_no_reduction() {
        System.out.println("test_no_reduction");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            // Test with logits.
            float[] true_np1 = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            BinaryCrossentropy instance = new BinaryCrossentropy(tf, true, 0.0F, Reduction.NONE);
            Operand loss = instance.call(y_true1, logits);
            Float[] expected = {0.F, 66.666664F};
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_label_smoothing() {
        System.out.println("test_label_smoothing");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float label_smoothing = 0.1f;
            float[] true_array = {1f, 0f, 1f};
            float[] logits_array = {100.0f, -100.0f, -100.0f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(1, 3)));
            Operand logits = tf.reshape(tf.constant(logits_array), tf.constant(Shape.of(1, 3)));

            BinaryCrossentropy instance = new BinaryCrossentropy(tf, true, label_smoothing);
            Operand loss = instance.call(y_true, logits);
            System.out.println(loss.asOutput().shape());
            float expected = (100.0F + 50.0F * label_smoothing) / 3.0F;
            testSession.evaluate(expected, loss);
        } catch (Exception expected) {

        }
    }

}
