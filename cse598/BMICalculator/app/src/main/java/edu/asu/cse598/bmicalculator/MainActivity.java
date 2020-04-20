package edu.asu.cse598.bmicalculator;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    public static final String EXTRA_MESSAGE = "com.example.myfirstapp.MESSAGE";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        EditText heightEditText = (EditText) findViewById(R.id.height_input);
        EditText weightEditText = (EditText) findViewById(R.id.weight_input);
        TextView label = (TextView) findViewById(R.id.label);
        TextView message = (TextView) findViewById(R.id.message);
    }

    /** Called when the user taps the Send button */
//    public void sendMessage(View view) {
//        Intent intent = new Intent(this, DisplayMessageActivity.class);
//        EditText editText = (EditText) findViewById(R.id.editText);
//        String message = editText.getText().toString();
//        intent.putExtra(EXTRA_MESSAGE, message);
//        startActivity(intent);
//    }

    /** Called when user taps call BMI API */
    public void callBmi(View view){

    }

    /** Called when user taps educate me */
    public void educateMe(View view){

    }

    public void setMessage(int label){

    }

}
