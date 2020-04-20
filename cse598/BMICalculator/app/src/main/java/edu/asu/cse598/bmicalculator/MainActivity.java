package edu.asu.cse598.bmicalculator;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;

import edu.asu.cse598.bmicalculator.rest.BMIApi;
import edu.asu.cse598.bmicalculator.rest.BMIApiClient;
import edu.asu.cse598.bmicalculator.rest.BMIResult;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    public static final String EXTRA_MESSAGE = "edu.asu.cse598.bmicalculator.MESSAGE";
    private static final String TAG = MainActivity.class.getSimpleName();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        EditText heightEditText = (EditText) findViewById(R.id.height_input);
        EditText weightEditText = (EditText) findViewById(R.id.weight_input);
        TextView label = (TextView) findViewById(R.id.label);
        TextView message = (TextView) findViewById(R.id.message);
    }

    public void callBmi(View view) {

        BMIApi bmiService = BMIApiClient.getClient().create(BMIApi.class);
        Call<BMIResult> call = bmiService.calculateBmi("60", "156");


        call.enqueue(new Callback<BMIResult>() {
            @Override
            public void onResponse(Call<BMIResult> call, Response<BMIResult> response) {

                //TODO display the values

                String bmi = response.body().getBmi();
                List<String> more = response.body().getMore();
                String risk = response.body().getRisk();

                Log.i(TAG, ">>> " + bmi);
                Log.i(TAG, ">>> " + more.toString());
                Log.i(TAG, ">>> " + risk);

                ((TextView) findViewById(R.id.label)).append(bmi);
            }

            @Override
            public void onFailure(Call<BMIResult> call, Throwable t) {
                Log.e(TAG, t.toString());
            }
        });


//        Intent intent = new Intent(this, DisplayMessageActivity.class);
//        EditText editText = (EditText) findViewById(R.id.editText);
//        String message = editText.getText().toString();
//        intent.putExtra(EXTRA_MESSAGE, message);
//        startActivity(intent);
    }

    /**
     * Called when user taps educate me
     */
    public void educateMe(View view) {

    }

    public void setMessage(int label) {

    }

}
