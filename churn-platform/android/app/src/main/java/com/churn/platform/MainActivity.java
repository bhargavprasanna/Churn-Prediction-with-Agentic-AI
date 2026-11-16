package com.churn.platform;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.*;

/** Thin native shell: loads the platform UI from the backend and supports CSV file upload. */
public class MainActivity extends Activity {
    private WebView web; private ValueCallback<Uri[]> chooser;
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        web = new WebView(this); setContentView(web);
        WebSettings s = web.getSettings(); s.setJavaScriptEnabled(true); s.setDomStorageEnabled(true);
        web.setWebViewClient(new WebViewClient());
        web.setWebChromeClient(new WebChromeClient() {
            @Override public boolean onShowFileChooser(WebView w, ValueCallback<Uri[]> cb, FileChooserParams p) {
                if (chooser != null) chooser.onReceiveValue(null);
                chooser = cb;
                Intent i = new Intent(Intent.ACTION_GET_CONTENT); i.addCategory(Intent.CATEGORY_OPENABLE); i.setType("*/*");
                startActivityForResult(Intent.createChooser(i, "Choose customers.csv"), 1); return true;
            }
        });
        web.loadUrl(BuildConfig.SERVER_URL);
    }
    @Override protected void onActivityResult(int rq, int rs, Intent d) {
        if (rq == 1 && chooser != null) { chooser.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(rs, d)); chooser = null; }
    }
    @Override public void onBackPressed() { if (web.canGoBack()) web.goBack(); else super.onBackPressed(); }
}
