from streamlit.testing.v1 import AppTest


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Kontakt aufnehmen" in c for c in captions)


def test_default_training_mode_is_zentriert():
    at = _fresh()
    radios = [r for r in at.radio if r.label == "Trainingsmodus"]
    assert radios[0].value == "zentriert"


def test_preset_buttons_exist_and_work():
    at = _fresh()
    labels = [b.label for b in at.button]
    assert "Nur zentriert trainiert" in labels
    assert "Überall trainiert (Kontrollgruppe)" in labels
    btn = [b for b in at.button if b.label == "Überall trainiert (Kontrollgruppe)"][0]
    btn.click().run()
    assert not at.exception


def test_k_slider_extreme_values_do_not_crash():
    at = _fresh()
    k_slider = [s for s in at.slider if s.label.startswith("Rastergröße")][0]
    k_slider.set_value(k_slider.min).run()
    assert not at.exception
    k_slider = [s for s in at.slider if s.label.startswith("Rastergröße")][0]
    k_slider.set_value(k_slider.max).run()
    assert not at.exception


def test_filters_slider_extreme_values_do_not_crash():
    at = _fresh()
    f_slider = [s for s in at.slider if s.label.startswith("Filter")][0]
    f_slider.set_value(f_slider.min).run()
    assert not at.exception
    f_slider = [s for s in at.slider if s.label.startswith("Filter")][0]
    f_slider.set_value(f_slider.max).run()
    assert not at.exception
