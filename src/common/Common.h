#ifndef COMMON_H
#define COMMON_H

#define PROPERTY(type, name) \
    private: \
        type m_##name; \
    public: \
        type name() const { return m_##name; } \
        void set##name(const type& _value) { m_##name = _value; }

#define PROPERTY_INIT(name, value) \
    m_##name(value)

#endif // COMMON_H
