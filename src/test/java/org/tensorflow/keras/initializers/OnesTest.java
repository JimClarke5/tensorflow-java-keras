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
package org.tensorflow.keras.initializers;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

/**
 * Test the Ones initializer
 */
public class OnesTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public OnesTest() {
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
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallUInt() {
        Byte[] expected = {1, 1, 1, 1}; // init to ones to make sure they all changet to zero
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            Ones<TUint8> instance = new Ones<>(tf);
            Operand<TUint8> operand = instance.call(tf.constant(shape), TUint8.DTYPE);
            session.evaluate(expected, operand);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallInt() {
        int[] expected = {1, 1, 1, 1}; // init to ones to make sure they all changet to zero
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            Ones<TInt32> instance = new Ones<>(tf);
            Operand<TInt32> operand = instance.call(tf.constant(shape), TInt32.DTYPE);
            session.evaluate(expected, operand);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallLong() {
        long[] expected = {1, 1, 1, 1}; // init to ones to make sure they all changet to zero
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            Ones<TInt64> instance = new Ones<>(tf);
            Operand<TInt64> operand = instance.call(tf.constant(shape), TInt64.DTYPE);
            session.evaluate(expected, operand);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallFloat() {
        float[] expected = {1.f, 1.f, 1.f, 1.f};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);
            Ones<TFloat32> instance = new Ones<>(tf);
            Operand<TFloat32> operand = instance.call(tf.constant(shape), TFloat32.DTYPE);
            session.evaluate(expected, operand);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallDouble() {
        double[] expected = {1., 1., 1., 1.};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);

            Ones<TFloat64> instance = new Ones<>(tf);
            Operand<TFloat64> operand = instance.call(tf.constant(shape), TFloat64.DTYPE);
            session.evaluate(expected, operand);
        }
    }

    /**
     * Test of call method, of class Ones.
     */
    @Test
    public void testCallString() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);

            Ones<TString> instance = new Ones<>(tf);
            Operand<TString> operand = instance.call(tf.constant(shape), TString.DTYPE);
            fail("AssertionError should have been thrown for TString");
        } catch (AssertionError expected) {
        }
    }

    @Test
    public void testCallBool() {
        Boolean[] expected = {true, true, true, true};
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Shape shape = Shape.of(2, 2);

            Ones<TBool> instance = new Ones<>(tf);
            Operand<TBool> operand = instance.call(tf.constant(shape), TBool.DTYPE);
            session.evaluate(expected, operand);
        }
    }

}
